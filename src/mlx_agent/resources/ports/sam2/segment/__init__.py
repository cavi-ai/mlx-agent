"""Run a prompted image mask on an explicitly selected local SAM 2 checkpoint."""

import hashlib
from pathlib import Path

VIDEO_ONLY = (
    'mask_downsample.', 'memory_attention.', 'memory_encoder.',
    'memory_temporal_positional_encoding', 'no_memory_positional_encoding',
    'no_object_pointer', 'object_pointer_proj.',
    'occlusion_spatial_embedding_parameter', 'temporal_positional_encoding_projection_layer.',
)


def validate_loading(info):
    """Reject incomplete image weights and retain only known video-specific extras."""
    fields = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
    if not isinstance(info, dict) or any(not isinstance(info.get(key), (list, tuple, set)) for key in fields):
        raise ValueError('The checkpoint loading report is absent or invalid.')
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('The checkpoint did not load all image model parameters.')
    unexpected = sorted(info.get('unexpected_keys', ()))
    if any(not any(key.startswith(prefix) for prefix in VIDEO_ONLY) for key in unexpected):
        raise ValueError('The checkpoint contains unexpected non-video parameters.')
    return unexpected


def load(path):
    import torch
    from transformers import Sam2Model, Sam2Processor
    if not torch.backends.mps.is_available():
        raise RuntimeError('The MPS device is unavailable.')
    processor = Sam2Processor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, info = Sam2Model.from_pretrained(
        path, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        torch_dtype=torch.float32, output_loading_info=True,
    )
    unexpected = validate_loading(info)
    return model.to('mps').eval(), processor, unexpected


def generate(loaded, image_path, output_path, box, points, labels, threshold):
    import torch
    from PIL import Image, ImageOps
    model, processor, unexpected = loaded
    source = Path(image_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
    width, height = image.size
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError('Box is outside the oriented image.')
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise ValueError('Point is outside the oriented image.')
    kwargs = {'images': image, 'input_boxes': [[box]], 'return_tensors': 'pt'}
    if points:
        kwargs.update(input_points=[[points]], input_labels=[[labels]])
    inputs = processor(**kwargs)
    sizes = inputs.pop('original_sizes')
    inputs = {key: value.to('mps') if hasattr(value, 'to') else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(output.pred_masks.cpu(), sizes, mask_threshold=threshold)
    if len(masks) != 1 or tuple(masks[0].shape) != (1, 1, height, width):
        raise ValueError('The mask output has an unexpected shape.')
    mask = masks[0][0, 0].to(dtype=torch.uint8).numpy() * 255
    Image.fromarray(mask, mode='L').save(output_path, format='PNG')
    return {'path': str(output_path), 'device': 'mps', 'input_sha256': digest,
            'dimensions': [width, height], 'threshold': threshold,
            'iou_score': float(output.iou_scores.reshape(-1)[0].cpu()),
            'unused_video_keys': unexpected,
            'memory': {'framework': 'torch', 'active_bytes': torch.mps.current_allocated_memory(),
                       'cache_bytes': None, 'peak_bytes': None}}
