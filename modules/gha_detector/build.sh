#!/usr/bin/env bash
# build.sh - Compile gha-ci-detector into a standalone single-file binary
# via PyInstaller, and install it into a user-chosen directory.
#
# Usage:
#   ./build.sh --install-dir PATH     Compile and install binary into PATH.
#   ./build.sh --install-dir PATH --force
#                                     Rebuild even if cache is valid.
#   ./build.sh --install-dir PATH --check
#                                     Only verify install is ready. No build.
#   ./build.sh --install-dir PATH --quiet
#                                     Suppress human logs (stderr).
#   ./build.sh -h | --help            Show this help.
#
# Output (stdout, single line JSON on success):
#   {"status":"ready","bin":"<install-dir>/gha-ci-detector","install_dir":"...",
#    "source_hash":"...","cached":true|false}
#
# Human logs go to stderr. Exit 0 on success, non-zero otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

INSTALL_DIR=""
FORCE=0
CHECK_ONLY=0
QUIET=0

show_help() {
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)
            [[ $# -ge 2 ]] || { echo "--install-dir needs a value" >&2; exit 2; }
            INSTALL_DIR="$2"
            shift 2
            ;;
        --install-dir=*)
            INSTALL_DIR="${1#*=}"
            shift
            ;;
        --force) FORCE=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --quiet) QUIET=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

log() {
    if [[ $QUIET -eq 0 ]]; then
        echo "[build] $*" >&2
    fi
}

die() {
    local msg="$1"
    echo "[build] ERROR: $msg" >&2
    printf '{"status":"error","message":"%s"}\n' "${msg//\"/\\\"}"
    exit 1
}

# --- Preconditions ---
[[ -n "$INSTALL_DIR" ]] || die "--install-dir is required (pass a directory path)"
command -v uv >/dev/null 2>&1 || die "uv not found in PATH"
[[ -f pyproject.toml ]] || die "pyproject.toml not found (wrong dir?)"
[[ -f scripts/entry.py ]] || die "scripts/entry.py missing"

# Normalize install dir to a clean absolute path (parent must exist).
case "$INSTALL_DIR" in
    /*) ;;
    *) INSTALL_DIR="$SCRIPT_DIR/$INSTALL_DIR" ;;
esac
mkdir -p "$INSTALL_DIR" || die "cannot create install dir: $INSTALL_DIR"
INSTALL_DIR="$(cd "$INSTALL_DIR" && pwd -P)"

BIN_NAME="gha-ci-detector"
BIN_PATH="$INSTALL_DIR/$BIN_NAME"
DIST_DIR="$SCRIPT_DIR/dist"
BUILD_DIR="$SCRIPT_DIR/build"
CACHE_DIR="$SCRIPT_DIR/.build-cache"
CACHE_MARKER="$CACHE_DIR/bin.sha256"

# --- Compute source hash (covers gha_ci_detector/, scripts/, pyproject.toml, uv.lock) ---
compute_hash() {
    {
        find gha_ci_detector scripts -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' \) -print0 2>/dev/null \
            | sort -z | xargs -0 sha256sum 2>/dev/null || true
        [[ -f pyproject.toml ]] && sha256sum pyproject.toml
        [[ -f uv.lock ]] && sha256sum uv.lock
    } | sha256sum | awk '{print $1}'
}

CURRENT_HASH=$(compute_hash)
CACHED_HASH=""
[[ -f "$CACHE_MARKER" ]] && CACHED_HASH=$(cat "$CACHE_MARKER")

# Smoke-test the installed binary: exits 0 and prints at least one line.
smoke_ok() {
    local bin="$1"
    [[ -x "$bin" ]] || return 1
    local out
    out=$("$bin" --version 2>/dev/null) || return 1
    [[ -n "$out" ]] || return 1
    return 0
}

emit_ready() {
    local cached="$1"
    printf '{"status":"ready","bin":"%s","install_dir":"%s","source_hash":"%s","cached":%s}\n' \
        "$BIN_PATH" "$INSTALL_DIR" "$CURRENT_HASH" "$cached"
}

# --- Check mode: verify readiness without building ---
if [[ $CHECK_ONLY -eq 1 ]]; then
    if [[ "$CURRENT_HASH" == "$CACHED_HASH" ]] && smoke_ok "$BIN_PATH"; then
        emit_ready true
        exit 0
    fi
    echo '{"status":"not_ready"}'
    exit 1
fi

BUILT_BIN="$DIST_DIR/$BIN_NAME"

# --- Cache hit: skip everything if install already satisfies current sources ---
if [[ $FORCE -eq 0 \
    && "$CURRENT_HASH" == "$CACHED_HASH" ]] && smoke_ok "$BIN_PATH"; then
    log "cache hit (hash=$CURRENT_HASH) — skipping build"
    emit_ready true
    exit 0
fi

# --- Reinstall-only path: binary already built for current sources, just copy. ---
if [[ $FORCE -eq 0 \
    && "$CURRENT_HASH" == "$CACHED_HASH" \
    && -x "$BUILT_BIN" ]] && smoke_ok "$BUILT_BIN"; then
    log "sources unchanged — reusing $BUILT_BIN (no recompile)"
    log "installing to $BIN_PATH"
    install -m 0755 "$BUILT_BIN" "$BIN_PATH"
    smoke_ok "$BIN_PATH" || die "installed binary failed smoke test: $BIN_PATH"
    emit_ready true
    exit 0
fi

# --- Full build ---
log "compiling $BIN_NAME via PyInstaller (install_dir=$INSTALL_DIR)"

# Sync deps. Prefer --frozen when lockfile exists and venv present.
if [[ -f uv.lock && -d .venv ]]; then
    log "uv sync --frozen"
    uv sync --frozen >&2
else
    log "uv sync"
    uv sync >&2
fi

# Clean previous PyInstaller outputs for this binary to avoid stale artifacts.
rm -rf "$BUILD_DIR/$BIN_NAME" "$DIST_DIR/$BIN_NAME"

log "pyinstaller --onefile"
uv run --with pyinstaller --with-editable . -- \
    pyinstaller \
        --onefile \
        --name "$BIN_NAME" \
        --paths . \
        --collect-all gha_ci_detector \
        --collect-all typer \
        --collect-all click \
        --collect-all rich \
        --collect-all ruamel \
        --collect-all yamllint \
        --collect-all tqdm \
        --noconfirm \
        --clean \
        --distpath "$DIST_DIR" \
        --workpath "$BUILD_DIR" \
        --specpath "$BUILD_DIR" \
        scripts/entry.py >&2

[[ -x "$BUILT_BIN" ]] || die "PyInstaller did not produce $BUILT_BIN"

# --- Install to user-chosen directory ---
log "installing to $BIN_PATH"
install -m 0755 "$BUILT_BIN" "$BIN_PATH"

# --- Verify ---
smoke_ok "$BIN_PATH" || die "installed binary failed smoke test: $BIN_PATH"

mkdir -p "$CACHE_DIR"
echo "$CURRENT_HASH" > "$CACHE_MARKER"

log "ready: $BIN_PATH"
emit_ready false
