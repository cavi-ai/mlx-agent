"""Bounded upstream reads for the loopback HTTP gateways."""

from dataclasses import dataclass
from http.client import HTTPResponse
import io
import math
import os
import selectors
import socket
import time

MAX_RELAY_SECONDS = 86400


class ClientDisconnected(OSError):
    pass


@dataclass(frozen=True)
class RelayLimits:
    headers: float = 600
    idle: float = 120
    total: float = 3600

    @classmethod
    def from_environment(cls):
        defaults = cls()
        values = {}
        for name in ("headers", "idle", "total"):
            key = "MLX_AGENT_RELAY_" + name.upper() + "_SECONDS"
            try:
                value = float(os.environ.get(key, getattr(defaults, name)))
                if not math.isfinite(value) or not 0 < value <= MAX_RELAY_SECONDS:
                    raise ValueError
            except ValueError:
                raise ValueError(f"{key} must be more than 0 and at most {MAX_RELAY_SECONDS} seconds.") from None
            values[name] = value
        return cls(**values)


def client_closed(sock):
    try:
        return not sock.recv(1, socket.MSG_PEEK)
    except OSError:
        return True


class Reply:
    """Use the standard HTTP parser with deadlines on its underlying reads."""

    def __init__(self, connection, limits, client=None):
        self.connection = connection
        self.socket = connection.sock
        self.client = client
        self.idle = limits.idle
        now = time.monotonic()
        self.deadline = now + limits.total
        self.headers_deadline = now + limits.headers
        self.socket.setblocking(False)
        connection.response_class = self._response

    def _response(self, sock, **kwargs):
        return HTTPResponse(self, **kwargs)

    def makefile(self, mode):
        return io.BufferedReader(_ReplyReader(self, self.socket.makefile(mode, buffering=0)))

    def getresponse(self):
        response = self.connection.getresponse()
        self.headers_deadline = None
        return response

    def wait_for_headers(self, client):
        with selectors.DefaultSelector() as selector:
            selector.register(self.socket, selectors.EVENT_READ)
            selector.register(client, selectors.EVENT_READ)
            while True:
                for key, _ in self.ready(selector, min(self.deadline, self.headers_deadline), client):
                    if key.fileobj is self.socket:
                        return

    @staticmethod
    def ready(selector, deadline, client):
        remaining = deadline - time.monotonic()
        events = selector.select(remaining) if remaining > 0 else []
        if not events:
            raise TimeoutError("The upstream reply timed out.")
        if client is not None and any(key.fileobj is client for key, _ in events):
            if client_closed(client):
                raise ClientDisconnected("The client disconnected.")
            # Pipelined input cannot be read as part of this request.
            selector.unregister(client)
        return events

    def drain(self, seconds):
        self.deadline = time.monotonic() + seconds
        self.headers_deadline = self.deadline
        self.idle = seconds


class _ReplyReader(io.RawIOBase):
    def __init__(self, reply, raw):
        self.reply, self.raw = reply, raw
        self.selector = selectors.DefaultSelector()
        self.selector.register(reply.socket, selectors.EVENT_READ)
        if reply.client is not None:
            self.selector.register(reply.client, selectors.EVENT_READ)

    def readable(self):
        return True

    def readinto(self, buffer):
        reply = self.reply
        deadline = min(reply.deadline, reply.headers_deadline if reply.headers_deadline is not None
                       else time.monotonic() + reply.idle)
        while True:
            events = reply.ready(self.selector, deadline, reply.client)
            if any(key.fileobj is reply.socket for key, _ in events):
                count = self.raw.readinto(buffer)
                if count is not None:
                    return count

    def close(self):
        try:
            self.selector.close()
        finally:
            self.raw.close()
            super().close()
