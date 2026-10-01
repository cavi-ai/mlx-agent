"""Synthetic backend manifests for unit tests; no real wheels involved."""


def synthetic_manifests():
    return {
        "mlx-lm": {
            "schema": "backend/1", "id": "mlx-lm", "version": "0.0-test", "builtin": True,
            "package": "mlx_lm", "convert": "mlx_lm.convert", "lock": None,
            "categories": {"text_llm": "models"},
            "remap_files": {"text_llm": ["utils.py"]},
            "registry": {"text_llm": {
                "model_types": ["gemma3", "glm", "llama", "mistral3", "qwen2", "qwen2_vl", "qwen3_moe"],
                "remapping": {"llava": "mistral3", "mistral": "llama"},
            }},
        },
        "mlx-vlm": {
            "schema": "backend/1", "id": "mlx-vlm", "version": "0.0-test", "builtin": False,
            "package": "mlx_vlm", "convert": "mlx_vlm.convert", "lock": "mlx-vlm.lock",
            "categories": {"vision_language": "models"},
            "remap_files": {"vision_language": ["utils.py"]},
            "registry": {"vision_language": {
                "model_types": ["glm", "glm4_moe_lite", "llama", "moondream3", "qwen2", "qwen2_vl"],
                "remapping": {}, "vision_types": ["moondream3", "qwen2_vl"],
            }},
        },
        "mlx-audio": {
            "schema": "backend/1", "id": "mlx-audio", "version": "0.0-test", "builtin": False,
            "package": "mlx_audio", "convert": "mlx_audio.convert", "lock": "mlx-audio.lock",
            "categories": {"speech_to_text": "stt/models", "text_to_speech": "tts/models"},
            "remap_files": {"speech_to_text": ["stt/utils.py"], "text_to_speech": ["tts/utils.py"]},
            "registry": {
                "speech_to_text": {
                    "model_types": ["glmasr", "voxtral", "voxtral_realtime", "whisper"],
                    "remapping": {"glm": "glmasr"},
                },
                "text_to_speech": {"model_types": ["kokoro"], "remapping": {}},
            },
        },
    }


def synthetic_registries(manifests, installed=()):
    return {
        backend_id: {"registry": manifest["registry"], "source": "snapshot", "installed": backend_id in installed}
        for backend_id, manifest in manifests.items()
    }
