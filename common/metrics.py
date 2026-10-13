"""Disk-size and GPU-memory measurement helpers."""
import json
import subprocess
from pathlib import Path


def dir_size_bytes(path):
    """Total size of weight/config files under `path` (follows nothing, counts regular files)."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
        elif p.is_symlink() and p.resolve().is_file():
            total += p.resolve().stat().st_size  # HF cache snapshots are symlinks into blobs/
    return total


def weights_size_bytes(path):
    """Size of weight files only (safetensors / bin), so tokenizer files don't muddy the comparison."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.suffix in (".safetensors", ".bin") and (p.is_file() or p.is_symlink()):
            total += p.resolve().stat().st_size
    return total


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    except ImportError:
        return {"available": False}


def pip_freeze():
    out = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True)
    return out.stdout.splitlines()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path
