"""Image references shared by local API request translators."""

import base64
import binascii
from urllib.parse import urlsplit

MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
DETAILS = {"auto", "low", "high", "original"}


def image_url(value, *, allow_data=True):
    """Validate a reference without fetching, decoding pixels or changing bytes."""
    if not isinstance(value, str) or not value:
        raise ValueError("Images need a data URL or an HTTP(S) URL.")
    if value.startswith("data:"):
        if not allow_data:
            raise ValueError("URL image sources must use HTTP(S).")
        header, separator, data = value.partition(",")
        if not separator or header not in {"data:" + mime + ";base64" for mime in MEDIA_TYPES} or not data:
            raise ValueError("Image data URLs need a supported media type and base64 data.")
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Image data must be valid base64.") from None
    else:
        try:
            parts = urlsplit(value)
            valid = (value.startswith(("http://", "https://")) and parts.hostname and parts.port != 0
                     and parts.username is None and parts.password is None
                     and not any(character.isspace() or ord(character) < 32 or ord(character) == 127
                                 for character in value))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Image URLs must use HTTP(S) without embedded credentials.")
    return {"type": "image_url", "image_url": {"url": value}}


def messages_image(block):
    source = block.get("source")
    if set(block) - {"type", "source"} or not isinstance(source, dict):
        raise ValueError("Image blocks need a supported source.")
    if source.get("type") == "base64" and set(source) == {"type", "media_type", "data"}:
        if (not isinstance(source["media_type"], str) or source["media_type"] not in MEDIA_TYPES
                or not isinstance(source["data"], str)):
            raise ValueError("Image sources need a supported media type and base64 data.")
        return image_url("data:" + source["media_type"] + ";base64," + source["data"])
    if source.get("type") == "url" and set(source) == {"type", "url"}:
        return image_url(source["url"], allow_data=False)
    raise ValueError("Image sources must use base64 data or an HTTP(S) URL.")


def responses_image(part):
    if set(part) - {"type", "image_url", "detail"}:
        raise ValueError("Image inputs support image_url and an optional detail hint.")
    value = image_url(part.get("image_url"))
    if "detail" in part:
        if not isinstance(part["detail"], str) or part["detail"] not in DETAILS:
            raise ValueError("Unsupported image detail hint.")
        value["image_url"]["detail"] = part["detail"]
    return value
