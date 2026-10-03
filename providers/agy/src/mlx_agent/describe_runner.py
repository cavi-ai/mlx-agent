"""Run inside a vision-language backend's environment: load one converted model, answer one question about an image or video, print JSON."""

import argparse
import json
import sys
import time


def _fit(image, max_pixels):
    """Shrink a PIL image, keeping its aspect ratio, until it holds at most max_pixels pixels."""
    width, height = image.size
    if max_pixels is None or width * height <= max_pixels:
        return image
    scale = (max_pixels / float(width * height)) ** 0.5
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))))


def _refuse(error, model_type, detail):
    print(json.dumps({"error": error, "model_type": model_type, "detail": detail}), flush=True)
    return 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--video")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=None)
    arguments = parser.parse_args(argv)

    import mlx.core as mx
    from mlx_vlm import apply_chat_template, generate, load
    from mlx_vlm.prompt_utils import MODEL_CONFIG

    started = time.time()
    model, processor = load(arguments.model)
    load_seconds = time.time() - started
    config = model.config
    model_type = str(getattr(config, "model_type", None) or "")

    kind = "image" if arguments.image else "video"
    if model_type.lower() not in MODEL_CONFIG:
        error = "not_vision_model" if kind == "image" else "video_unsupported"
        return _refuse(error, model_type, "mlx_vlm has no chat format for images or frames of this model type")

    images, videos, template = [], None, {}
    if kind == "image":
        images = [arguments.image]
    else:
        from mlx_vlm.generate.video import processor_handles_video, resolve_video_inputs
        from mlx_vlm.utils import VideoSampling

        if processor_handles_video(processor):
            videos = [arguments.video]
            template = {"video": videos, "fps": arguments.fps}
            if arguments.max_pixels is not None:
                template["max_pixels"] = arguments.max_pixels
        else:
            try:
                resolution = resolve_video_inputs(
                    processor, [arguments.video], fps=arguments.fps, max_frames=16,
                    sampling=VideoSampling(fps=arguments.fps),
                )
            except ImportError as error:
                return _refuse("video_unsupported", model_type, "frame sampling needs a video decoder: {0}".format(error))
            images = list(resolution.images)
    if arguments.max_pixels is not None and images:
        from PIL import Image

        images = [_fit(Image.open(item) if isinstance(item, str) else item, arguments.max_pixels) for item in images]

    prompt = apply_chat_template(processor, config, arguments.prompt, num_images=len(images), **template)

    kwargs = {"max_tokens": arguments.max_tokens, "temperature": arguments.temperature, "verbose": False}
    if videos:
        kwargs["video"] = videos
        kwargs["fps"] = arguments.fps
    if images:
        kwargs["image"] = images

    mx.reset_peak_memory()
    started = time.time()
    result = generate(model, processor, prompt, **kwargs)
    seconds = time.time() - started

    print(json.dumps({
        "text": str(result.text).strip(),
        "prompt_tokens": int(result.prompt_tokens),
        "generation_tokens": int(result.generation_tokens),
        "prompt_tps": round(float(result.prompt_tps), 3),
        "generation_tps": round(float(result.generation_tps), 3),
        "peak_memory_gb": round(float(result.peak_memory), 3),
        "seconds": round(seconds, 3),
        "load_seconds": round(load_seconds, 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
