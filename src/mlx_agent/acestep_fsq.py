"""Create ACE-Step's quantizer constants outside Transformers' meta device."""
import torch
from vector_quantize_pytorch import ResidualFSQ as _ResidualFSQ


class ResidualFSQ(_ResidualFSQ):
    def __init__(self, *args, **kwargs):
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
