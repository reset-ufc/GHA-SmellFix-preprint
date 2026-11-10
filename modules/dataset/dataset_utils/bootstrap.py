"""Ensure shipped artifacts are present locally; verify + extract on demand.

Used by notebooks/scripts that consume large artifacts shipped as `.tar.xz`
alongside a `.sha256.txt` sidecar. Idempotent: skips work if target already
extracted.
"""

from __future__ import annotations

import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

from dataset_utils.hashing import compute_file_hash, read_sha256_sidecar


class BootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class Artifact:
    archive: Path
    sidecar: Path
    extract_to: Path
    marker: Path


def _archive_for(target_dir: Path) -> Artifact:
    archive = target_dir.with_suffix(target_dir.suffix + ".tar.xz")
    sidecar = archive.with_suffix(archive.suffix + ".sha256.txt")
    return Artifact(
        archive=archive,
        sidecar=sidecar,
        extract_to=target_dir.parent,
        marker=target_dir,
    )


def verify_archive(archive: Path, sidecar: Path) -> str:
    if not archive.exists():
        msg = f"Archive missing: {archive}"
        raise BootstrapError(msg)
    if not sidecar.exists():
        msg = f"Sidecar SHA file missing: {sidecar}"
        raise BootstrapError(msg)
    expected = read_sha256_sidecar(sidecar)
    actual = compute_file_hash(archive)
    if actual != expected:
        msg = f"SHA-256 mismatch for {archive.name}:\n  expected: {expected}\n  actual  : {actual}"
        raise BootstrapError(msg)
    return actual


def extract_archive(archive: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                msg = f"Refusing unsafe member path in archive: {name}"
                raise BootstrapError(msg)
        tar.extractall(extract_to, filter="data")


def ensure_extracted(target_dir: Path, *, force: bool = False, verbose: bool = True) -> Path:
    """Ensure `target_dir` exists locally; extract from sibling `.tar.xz` if missing.

    Layout assumption — for `target_dir = data/downloaded_workflows`:
      - `data/downloaded_workflows.tar.xz`         (archive)
      - `data/downloaded_workflows.tar.xz.sha256.txt`  (sidecar)
      - `data/downloaded_workflows/`               (extracted target)
    """
    artifact = _archive_for(target_dir)

    if target_dir.exists() and not force:
        if verbose:
            print(f"[bootstrap] {target_dir} already present — skipping extract.")
        return target_dir

    if verbose:
        print(f"[bootstrap] verifying {artifact.archive.name}...", file=sys.stderr)
    digest = verify_archive(artifact.archive, artifact.sidecar)
    if verbose:
        print(f"[bootstrap] sha256 ok ({digest[:12]}...)", file=sys.stderr)

    if force and target_dir.exists():
        if verbose:
            print(f"[bootstrap] force=True → removing existing {target_dir}", file=sys.stderr)
        import shutil

        shutil.rmtree(target_dir)

    if verbose:
        print(f"[bootstrap] extracting → {artifact.extract_to}", file=sys.stderr)
    extract_archive(artifact.archive, artifact.extract_to)

    if not target_dir.exists():
        msg = (
            f"Extraction did not produce expected dir: {target_dir}. "
            f"Archive {artifact.archive} may have wrong top-level structure."
        )
        raise BootstrapError(msg)

    if verbose:
        print(f"[bootstrap] ready: {target_dir}", file=sys.stderr)
    return target_dir


def _without_owner(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Record every entry as owned by 0/0 with no user or group name."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def package_dir(source_dir: Path, *, replace: bool = True) -> tuple[Path, Path]:
    """Compress `source_dir` into sibling `.tar.xz` + write `.sha256.txt` sidecar.

    The entries do not record the user or group of the machine that packaged them.
    Returns (archive_path, sidecar_path).
    """
    if not source_dir.is_dir():
        msg = f"package_dir: not a directory: {source_dir}"
        raise BootstrapError(msg)
    artifact = _archive_for(source_dir)
    if artifact.archive.exists():
        if not replace:
            msg = f"Archive already exists: {artifact.archive} (pass replace=True to overwrite)"
            raise BootstrapError(msg)
        artifact.archive.unlink()
    artifact.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(artifact.archive, "w:xz") as tar:
        tar.add(source_dir, arcname=source_dir.name, filter=_without_owner)
    digest = compute_file_hash(artifact.archive)
    artifact.sidecar.write_text(f"{digest}  {artifact.archive.name}\n", encoding="utf-8")
    return artifact.archive, artifact.sidecar
