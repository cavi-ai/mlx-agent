import unittest

from mlx_agent import messages_api, responses_api
from mlx_agent.probe_assets import vision_probe_image_base64


IMAGE = "data:image/png;base64," + vision_probe_image_base64()
URL = "https://example.invalid/second.png"


class ImageRequestTests(unittest.TestCase):
    def test_messages_preserves_ordered_text_base64_and_url_images(self):
        body = {"model": "local", "max_tokens": 64, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Compare "},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": vision_probe_image_base64()}},
            {"type": "text", "text": "with "},
            {"type": "image", "source": {"type": "url", "url": URL}}]}]}
        value = messages_api.to_chat_request(body, allow_images=True)
        self.assertEqual(value["messages"], [{"role": "user", "content": [
            {"type": "text", "text": "Compare "}, {"type": "image_url", "image_url": {"url": IMAGE}},
            {"type": "text", "text": "with "}, {"type": "image_url", "image_url": {"url": URL}}]}])

    def test_responses_preserves_ordered_images_and_detail_hints(self):
        for detail in (None, "auto", "low", "high", "original"):
            part = {"type": "input_image", "image_url": IMAGE}
            if detail is not None:
                part["detail"] = detail
            body = {"model": "local", "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Compare "}, part,
                {"type": "input_text", "text": "with "}, {"type": "input_image", "image_url": URL}]}]}
            image = {"url": IMAGE}
            if detail is not None:
                image["detail"] = detail
            with self.subTest(detail=detail):
                value = responses_api.to_chat_request(body, allow_images=True)
                self.assertEqual(value["messages"], [{"role": "user", "content": [
                    {"type": "text", "text": "Compare "}, {"type": "image_url", "image_url": image},
                    {"type": "text", "text": "with "}, {"type": "image_url", "image_url": {"url": URL}}]}])

    def test_plain_text_keeps_its_existing_worker_shape_when_images_are_enabled(self):
        messages = {"model": "local", "max_tokens": 64, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]}
        responses = {"model": "local", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "one"}, {"type": "input_text", "text": "two"}]}]}
        for module, body in ((messages_api, messages), (responses_api, responses)):
            self.assertEqual(module.to_chat_request(body, allow_images=True), module.to_chat_request(body))

    def test_malformed_images_and_unsupported_roles_are_refused(self):
        images = ("", "file:///tmp/image.png", "/tmp/image.png", "https://", "https://user:pass@example.invalid/i.png",
                  "data:text/plain;base64,SGk=", "data:image/png,raw", "data:image/png;base64,!!",
                  "data:image/png;base64,")
        for image in images:
            with self.subTest(image=image), self.assertRaises(responses_api.ResponsesRequestError):
                responses_api.to_chat_request({"input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": image}]}]}, allow_images=True)
        for role in ("assistant", "system", "developer"):
            with self.subTest(role=role), self.assertRaises(responses_api.ResponsesRequestError):
                responses_api.to_chat_request({"input": [{"role": role, "content": [
                    {"type": "input_image", "image_url": IMAGE}]}]}, allow_images=True)
        for source in ({"type": "file", "file_id": "file_1"}, {"type": "base64", "media_type": "image/png", "data": "!"},
                       {"type": "base64", "media_type": [], "data": "AA=="},
                       {"type": "base64", "media_type": {}, "data": "AA=="},
                       {"type": "url", "url": URL, "unknown": True}):
            with self.subTest(source=source), self.assertRaises(messages_api.MessagesRequestError):
                messages_api.to_chat_request({"max_tokens": 64, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": source}]}]}, allow_images=True)
        for part in ({"type": "input_image", "file_id": "file_1"},
                     {"type": "input_image", "image_url": IMAGE, "detail": "invalid"},
                     {"type": "input_image", "image_url": IMAGE, "unknown": True}):
            with self.subTest(part=part), self.assertRaises(responses_api.ResponsesRequestError):
                responses_api.to_chat_request({"input": [{"role": "user", "content": [part]}]}, allow_images=True)
