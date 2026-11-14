import unittest

from mlx_agent.jit_serve import GatewayError, GatewayHandler
from mlx_agent.task_serve import TaskHandler


class _Controller:
    def unload(self):
        return {"unloaded": True}

    def reclaim_cache(self):
        return {"reclaimed": True}

    def configure(self, body):
        return {"policy": body}


class _Handler:
    def __init__(self, path):
        self.path = path
        self.server = type("Server", (), {"controller": _Controller()})()
        self.sent = None

    def _json(self, status, body):
        self.sent = (status, body)


class WeightControlTests(unittest.TestCase):
    def test_chat_and_task_handlers_share_one_control_dispatch(self):
        self.assertIs(TaskHandler._dispatch_control, GatewayHandler._dispatch_control)

    def test_unload_cache_and_policy_are_one_dispatch(self):
        unload = _Handler("/_mlx/unload")
        self.assertTrue(GatewayHandler._dispatch_control(unload, {}))
        self.assertEqual(unload.sent, (200, {"unloaded": True}))

        cache = _Handler("/_mlx/cache")
        self.assertTrue(GatewayHandler._dispatch_control(cache, {}))
        self.assertEqual(cache.sent, (200, {"reclaimed": True}))
        with self.assertRaises(GatewayError) as error:
            GatewayHandler._dispatch_control(_Handler("/_mlx/cache"), {"reclaim": True})
        self.assertEqual(error.exception.code, "invalid_body")

        policy = _Handler("/_mlx/policy")
        self.assertTrue(GatewayHandler._dispatch_control(policy, {"keep_loaded": True}))
        self.assertEqual(policy.sent, (200, {"policy": {"keep_loaded": True}}))
        self.assertFalse(GatewayHandler._dispatch_control(_Handler("/v1/chat/completions"), {}))
