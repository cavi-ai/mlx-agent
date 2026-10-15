"""Read a completed training receipt and recheck its local artifacts."""
import hashlib
import json
from pathlib import Path


def read_trained_adapter(receipt_path):
    from .jit_serve import model_fingerprint
    from .lora import adapter_identity
    from .wiring import _MODEL

    path = Path(receipt_path).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Choose a bounded local training receipt.")
    with path.open("rb") as stream:
        payload = stream.read(65537)
    if len(payload) > 65536 or path.is_symlink():
        raise ValueError("Choose a bounded local training receipt.")
    receipt = json.loads(payload)
    if (not isinstance(receipt, dict) or receipt.get("kind") != "lora"
            or receipt.get("exit_status") != "done" or type(receipt.get("pid")) is not int
            or not isinstance(receipt.get("argv"), list)):
        raise ValueError("A completed training receipt with verified adapter output is required.")
    repo = receipt.get("repo")
    if not isinstance(repo, str) or not _MODEL.fullmatch(repo):
        raise ValueError("The training receipt has no valid base model identifier.")
    output = receipt.get("out")
    if not isinstance(output, str) or not output:
        raise ValueError("The training receipt has no adapter output path.")
    adapter = Path(output).expanduser().absolute()
    identity = adapter_identity(adapter)
    if not identity or identity != receipt.get("adapter_identity"):
        raise ValueError("Adapter content no longer matches the completed training receipt.")
    base = receipt.get("base_identity")
    if not isinstance(base, dict) or not isinstance(base.get("path"), str) or not base["path"]:
        raise ValueError("This historical receipt has no pinned base identity; run a newly reviewed training job.")
    model = Path(base["path"]).expanduser().absolute()
    if not model.is_dir() or model_fingerprint(model) != base.get("fingerprint"):
        raise ValueError("The trained base-model file identity changed.")
    binding = {"training_receipt": str(path), "training_receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "training_run_id": receipt.get("run_id"), "base_repo": repo, "base_path": str(model),
        "base_fingerprint": base["fingerprint"], "adapter_path": str(adapter), "adapter_identity": identity}
    return receipt, binding
