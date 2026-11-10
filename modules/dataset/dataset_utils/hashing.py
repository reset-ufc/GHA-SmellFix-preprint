from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def write_sha256_sidecar(path: Path, digest: str | None = None) -> Path:
    if digest is None:
        digest = compute_file_hash(path)
    sidecar = path.with_suffix(path.suffix + ".sha256.txt")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return sidecar


def read_sha256_sidecar(sidecar: Path) -> str:
    text = sidecar.read_text(encoding="utf-8").strip()
    return text.split()[0] if text else ""
