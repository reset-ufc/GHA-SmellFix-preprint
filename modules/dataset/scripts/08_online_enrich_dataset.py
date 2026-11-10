"""Step 8: enrich filtered dataset with GitHub API data.

Uses multiple GitHub tokens with rotation, rate limit handling,
HTTP-level caching, persistent workflow corpus, and checkpoint/resume support.

Enriches each repository with:
- repo_is_fork
- repo_is_archived
- repo_default_branch

Enriches each workflow with:
- workflow_triggers (JSON)
- workflow_run_count

Applies criteria:
- EC1: exclude forked repositories
- EC2: exclude archived repositories
- IC7: keep workflows triggered by push or pull_request
- IC8: keep workflows with >= 100 runs

Two-tier on-disk layout:
- `data/downloaded_workflows/`  Persistent YAML corpus (consumed by Step 10/11; shipped as `.tar.xz`).
- `data/.http_cache/`           Ephemeral HTTP response cache for resume.
                                Gitignored; can be deleted after a successful run.
- `data/.checkpoint_enrich.json` and `data/8_dataset_enriched_raw.csv`
                                Resume checkpoint + raw output. Gitignored.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import yaml
from dataset_utils.paths import data_dir
from tqdm import tqdm

DATA_DIR = data_dir()

CHECKPOINT_FILE = DATA_DIR / ".checkpoint_enrich.json"
HTTP_CACHE_DIR = DATA_DIR / ".http_cache"
CORPUS_DIR = DATA_DIR / "downloaded_workflows"
LOG_DIR = DATA_DIR / ".logs"
LOG_FILE = LOG_DIR / "enrichment.log"
LOG_JSON_FILE = LOG_DIR / "enrichment.jsonl"

RAW_CSV = DATA_DIR / "8_dataset_enriched_raw.csv"
FILTERED_CSV = DATA_DIR / "8_dataset_enriched.csv"

DEFAULT_INPUT_CSV = DATA_DIR / "7_dataset_filtered.csv"
DEFAULT_OUTPUT_CSV = FILTERED_CSV
DEFAULT_PRIOR_REPORT = DATA_DIR / "step7_report.json"
DEFAULT_MINING_CONFIG = DATA_DIR / "mining_config.json"

_MSG_NO_HEADER = "Missing CSV header"
_MSG_NO_TOKENS = "No GitHub tokens found in config or GITHUB_TOKEN env var"

CRITERIA_KEYS = [
    "min_stars",
    "min_commits",
    "min_runs",
    "exclude_forks",
    "exclude_archived",
]

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class TokenState:
    token: str
    owner: str
    remaining: int = 5000
    reset_time: int = 0
    cooldown: bool = False


@dataclass
class EnrichStats:
    total_repos: int = 0
    repos_processed: int = 0
    repos_excluded_fork: int = 0
    repos_excluded_archived: int = 0
    repos_skipped_not_found: int = 0
    workflows_enriched: int = 0
    workflows_excluded_no_push_pr: int = 0
    workflows_excluded_low_runs: int = 0
    final_rows: int = 0
    api_calls: int = 0
    api_errors: int = 0
    token_rotations: int = 0
    freezes: int = 0
    cache_hits: int = 0
    started_at: str = ""
    finished_at: str = ""


class TokenPool:
    def __init__(self, tokens: list[dict], safety_threshold: int = 15):
        self.tokens = [TokenState(token=t["token"], owner=t["owner"]) for t in tokens]
        self.safety_threshold = safety_threshold
        self.current_index = 0

    def get_token(self) -> TokenState | None:
        for _ in range(len(self.tokens)):
            ts = self.tokens[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.tokens)
            if not ts.cooldown and ts.remaining > self.safety_threshold:
                return ts
        return None

    def mark_cooldown(self, ts: TokenState, reset_time: int = 0):
        ts.cooldown = True
        if reset_time:
            ts.reset_time = reset_time

    def reset_cooldowns(self):
        # Any token past its reset_time is refreshed regardless of cooldown flag
        # — low-budget tokens (remaining <= safety_threshold) also need refresh.
        now = int(time.time())
        for ts in self.tokens:
            if ts.reset_time and now >= ts.reset_time:
                ts.cooldown = False
                ts.remaining = 5000

    def time_to_next_reset(self) -> int:
        now = int(time.time())
        # Include tokens that are unusable for any reason (cooldown OR low budget),
        # so _acquire_token_async knows when ANY token will become usable again.
        resets = [
            ts.reset_time
            for ts in self.tokens
            if (ts.cooldown or ts.remaining <= self.safety_threshold) and ts.reset_time > now
        ]
        if not resets:
            return 0
        return min(resets) - now

    def status(self) -> list[dict]:
        return [
            {
                "owner": ts.owner,
                "remaining": ts.remaining,
                "cooldown": ts.cooldown,
                "reset_time": datetime.fromtimestamp(ts.reset_time).isoformat()
                if ts.reset_time
                else None,
            }
            for ts in self.tokens
        ]


class GitHubCache:
    """Two-tier store: ephemeral HTTP cache + persistent YAML corpus.

    HTTP cache (`http_cache_dir/{repos,workflows,runs}/`) is internal to Step 8
    and may be discarded after a successful run. The YAML corpus
    (`corpus_dir/`) is the persistent input to Steps 10 and 11 and is what gets
    shipped to reproducers via `downloaded_workflows.tar.xz`.
    """

    def __init__(self, http_cache_dir: Path, corpus_dir: Path):
        self.repos_dir = http_cache_dir / "repos"
        self.workflows_dir = http_cache_dir / "workflows"
        self.runs_dir = http_cache_dir / "runs"
        self.yaml_dir = corpus_dir
        for d in (self.repos_dir, self.workflows_dir, self.runs_dir, self.yaml_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _path(self, base_dir: Path, key: str) -> Path:
        return base_dir / f"{key.replace('/', '_')}.json"

    def get(self, key: str, category: str) -> dict | None:
        dirs = {
            "repos": self.repos_dir,
            "workflows": self.workflows_dir,
            "runs": self.runs_dir,
        }
        p = self._path(dirs[category], key)
        if p.exists():
            with p.open("r") as f:
                return json.load(f)
        return None

    def set(self, key: str, category: str, data: dict):
        dirs = {
            "repos": self.repos_dir,
            "workflows": self.workflows_dir,
            "runs": self.runs_dir,
        }
        p = self._path(dirs[category], key)
        with p.open("w") as f:
            json.dump(data, f)

    def _yaml_path(self, repo_slug: str, wf_path: str) -> Path:
        owner, repo = repo_slug.split("/", 1)
        filename = wf_path.split("/")[-1]
        safe_name = f"{owner}@{repo}@{filename}"
        return self.yaml_dir / safe_name

    def save_yaml(self, repo_slug: str, wf_path: str, content: str) -> Path:
        p = self._yaml_path(repo_slug, wf_path)
        with p.open("w") as f:
            f.write(content)
        return p

    def load_yaml(self, repo_slug: str, wf_path: str) -> str | None:
        p = self._yaml_path(repo_slug, wf_path)
        if p.exists():
            with p.open("r") as f:
                return f.read()
        return None


def setup_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("step8")
    logger.setLevel(logging.DEBUG)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    json_handler = logging.FileHandler(LOG_JSON_FILE, mode="a")
    json_handler.setLevel(logging.DEBUG)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            }
            if hasattr(record, "extra"):
                log_entry.update(record.extra)
            if record.exc_info and record.exc_info[0]:
                log_entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_entry, ensure_ascii=False)

    json_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_handler)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    if verbose:
        console_handler.setLevel(logging.DEBUG)

    return logger


def _make_client(token: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        timeout=30,
        http2=True,
    )


async def github_request(
    client: httpx.AsyncClient,
    url: str,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState | None = None,
    pool: TokenPool | None = None,
    max_network_retries: int = 4,
    max_rate_retries: int = 15,
) -> tuple[dict, dict]:
    # HTTP/2 stream resets + 403/429 rate-limit responses are both common on
    # long runs. Outer loop rotates the token on rate-limit hits; inner loop
    # retries on transient network errors.
    request_headers = {"Authorization": f"token {token_state.token}"} if token_state else None
    for rate_attempt in range(max_rate_retries):
        stats.api_calls += 1
        resp = None
        for attempt in range(max_network_retries):
            try:
                resp = await client.get(url, headers=request_headers)
                break
            except (
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.PoolTimeout,
            ) as e:
                stats.api_errors += 1
                logger.warning(
                    "Network error on %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    max_network_retries,
                    e,
                    extra={"event": "network_error", "url": url, "attempt": attempt + 1},
                )
                if attempt >= max_network_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
        try:
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            # Only rotate on real rate limits — x-ratelimit-remaining=0 or 429.
            # Permission-denied 403s (private repos, insufficient scope) stay raised,
            # otherwise they'd burn every token in an infinite cooldown loop.
            remaining_hdr = e.response.headers.get("x-ratelimit-remaining", "")
            is_rate_limit = e.response.status_code == 429 or (
                e.response.status_code == 403 and remaining_hdr == "0"
            )
            if is_rate_limit and pool is not None and token_state is not None:
                reset = int(e.response.headers.get("x-ratelimit-reset", 0))
                pool.mark_cooldown(token_state, reset)
                stats.token_rotations += 1
                logger.warning(
                    "Rate limit (%d) on %s; rotating token (attempt %d/%d)",
                    e.response.status_code,
                    url,
                    rate_attempt + 1,
                    max_rate_retries,
                    extra={
                        "event": "rate_limited",
                        "url": url,
                        "status": e.response.status_code,
                    },
                )
                token_state = await _acquire_token_async(pool, stats, logger)
                request_headers = {"Authorization": f"token {token_state.token}"}
                continue
            raise
    else:
        raise RuntimeError(f"Exhausted rate-limit retries ({max_rate_retries}) for {url}")
    data = resp.json()
    headers = dict(resp.headers)
    remaining_header = headers.get("x-ratelimit-remaining")
    remaining = int(remaining_header) if remaining_header and remaining_header.isdigit() else -1
    if token_state is not None and remaining >= 0:
        token_state.remaining = remaining
        reset_header = headers.get("x-ratelimit-reset")
        if reset_header and reset_header.isdigit():
            token_state.reset_time = int(reset_header)
            token_state.cooldown = False
    logger.debug(
        f"API GET {url} -> 200 (remaining={remaining})",
        extra={"event": "api_success", "url": url, "status": 200, "remaining": remaining},
    )
    return data, headers


async def fetch_repo_info(
    client: httpx.AsyncClient,
    repo_slug: str,
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState,
    pool: TokenPool | None = None,
) -> dict | None:
    cached = cache.get(repo_slug, "repos")
    if cached:
        stats.cache_hits += 1
        logger.debug(
            f"Cache hit: repo info for {repo_slug}",
            extra={"event": "cache_hit", "category": "repos", "repo": repo_slug},
        )
        return cached

    url = f"https://api.github.com/repos/{repo_slug}"
    try:
        data, _headers = await github_request(
            client, url, stats, logger, token_state=token_state, pool=pool
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (403, 404, 451):
            logger.warning(
                f"Repo not found: {repo_slug} (status {e.response.status_code})",
                extra={
                    "event": "repo_not_found",
                    "repo": repo_slug,
                    "status": e.response.status_code,
                },
            )
            cache.set(
                repo_slug,
                "repos",
                {"not_found": True, "status": e.response.status_code},
            )
            return None
        raise
    else:
        result = {
            "is_fork": data.get("fork", False),
            "is_archived": data.get("archived", False),
            "default_branch": data.get("default_branch", ""),
        }
        cache.set(repo_slug, "repos", result)
        logger.debug(
            f"Fetched repo info: {repo_slug}"
            f" (fork={result['is_fork']}, archived={result['is_archived']})",
            extra={"event": "repo_fetched", "repo": repo_slug, **result},
        )
        return result


async def fetch_workflows(
    client: httpx.AsyncClient,
    repo_slug: str,
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState,
    pool: TokenPool | None = None,
) -> list[dict]:
    cached = cache.get(repo_slug, "workflows")
    if cached:
        stats.cache_hits += 1
        logger.debug(
            f"Cache hit: workflows for {repo_slug}",
            extra={"event": "cache_hit", "category": "workflows", "repo": repo_slug},
        )
        return cached

    workflows = await _fetch_workflows_batch(
        client, repo_slug, cache, stats, logger, token_state, pool=pool
    )
    if workflows is None:
        url = f"https://api.github.com/repos/{repo_slug}/actions/workflows?per_page=100"
        try:
            data, _ = await github_request(
                client, url, stats, logger, token_state=token_state, pool=pool
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 451):
                logger.warning(
                    f"Workflows endpoint not found for {repo_slug} (status {e.response.status_code})",
                    extra={
                        "event": "workflows_not_found",
                        "repo": repo_slug,
                        "status": e.response.status_code,
                    },
                )
                return []
            raise
        else:
            workflows = []
            for wf in data.get("workflows", []):
                triggers = await _extract_triggers_from_repo_content(
                    client, repo_slug, wf["path"], cache, stats, logger, token_state, pool=pool
                )
                workflows.append(
                    {
                        "id": wf["id"],
                        "name": wf["name"],
                        "path": wf["path"],
                        "state": wf.get("state", "active"),
                        "created_at": wf.get("created_at", ""),
                        "updated_at": wf.get("updated_at", ""),
                        "triggers": triggers,
                    }
                )

    cache.set(repo_slug, "workflows", workflows)
    logger.debug(
        f"Fetched {len(workflows)} workflows for {repo_slug}",
        extra={
            "event": "workflows_fetched",
            "repo": repo_slug,
            "count": len(workflows),
        },
    )
    return workflows


async def _fetch_workflows_batch(
    client: httpx.AsyncClient,
    repo_slug: str,
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState,
    pool: TokenPool | None = None,
) -> list[dict] | None:
    url = f"https://api.github.com/repos/{repo_slug}/contents/.github/workflows"
    try:
        data, _ = await github_request(
            client, url, stats, logger, token_state=token_state, pool=pool
        )
    except httpx.HTTPStatusError:
        return None

    if not isinstance(data, list):
        return None

    workflows = []
    for wf_file in data:
        if not wf_file.get("name", "").endswith((".yml", ".yaml")):
            continue

        wf_path = wf_file.get("path", "")
        content = None

        cached = cache.load_yaml(repo_slug, wf_path)
        if cached is not None:
            stats.cache_hits += 1
            content = cached
        else:
            content_b64 = wf_file.get("content", "")
            if not content_b64:
                continue
            try:
                content = base64.b64decode(content_b64).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to decode workflow content for {repo_slug}/{wf_path}: {e}")
                continue
            cache.save_yaml(repo_slug, wf_path, content)

        triggers = _parse_triggers_from_content(content)
        workflows.append(
            {
                "id": wf_file.get("sha", ""),
                "name": wf_file.get("name", ""),
                "path": wf_path,
                "state": "active",
                "created_at": "",
                "updated_at": "",
                "triggers": triggers,
            }
        )

    return workflows if workflows else None


def _parse_triggers_from_parsed(parsed: dict) -> list[str]:
    on_field = parsed.get("on") or parsed.get(True)
    if isinstance(on_field, str):
        return [on_field]
    if isinstance(on_field, list):
        return [str(x) for x in on_field]
    if isinstance(on_field, dict):
        return list(on_field.keys())
    return []


def _parse_triggers_from_content(content: str) -> list[str]:
    try:
        parsed = yaml.safe_load(content)
    except Exception:
        # yaml.safe_load can raise ValueError (e.g. invalid date like 0000-00-00)
        # in addition to yaml.YAMLError — catch all parse failures.
        return []

    if not isinstance(parsed, dict):
        return []

    return _parse_triggers_from_parsed(parsed)


def _get_triggers_from_cached(
    cache: GitHubCache | None,
    repo_slug: str,
    wf_path: str,
    stats: EnrichStats,
) -> list[str] | None:
    if cache is None:
        return None
    cached = cache.load_yaml(repo_slug, wf_path)
    if cached is None:
        return None
    stats.cache_hits += 1
    try:
        parsed = yaml.safe_load(cached)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    return _parse_triggers_from_parsed(parsed)


async def _extract_triggers_from_repo_content(
    client: httpx.AsyncClient,
    repo_slug: str,
    wf_path: str,
    cache: GitHubCache | None,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState,
    pool: TokenPool | None = None,
) -> list[str]:
    cached_triggers = _get_triggers_from_cached(cache, repo_slug, wf_path, stats)
    if cached_triggers is not None:
        return cached_triggers

    url = f"https://api.github.com/repos/{repo_slug}/contents/{wf_path}"
    try:
        wf_data, _ = await github_request(
            client, url, stats, logger, token_state=token_state, pool=pool
        )
    except httpx.HTTPStatusError:
        return []

    if not isinstance(wf_data, dict):
        return []

    content_b64 = wf_data.get("content", "")
    try:
        content = base64.b64decode(content_b64).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to decode workflow content for {repo_slug}/{wf_path}: {e}")
        return []

    if cache is not None:
        cache.save_yaml(repo_slug, wf_path, content)

    try:
        parsed = yaml.safe_load(content)
    except Exception:
        # yaml.safe_load can raise ValueError (e.g. invalid date like 0000-00-00)
        # in addition to yaml.YAMLError — catch all parse failures.
        return []

    if not isinstance(parsed, dict):
        return []

    return _parse_triggers_from_parsed(parsed)


async def fetch_run_count(
    client: httpx.AsyncClient,
    repo_slug: str,
    workflow_id: int,
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState,
    pool: TokenPool | None = None,
) -> int:
    cache_key = f"{repo_slug}_{workflow_id}"
    cached = cache.get(cache_key, "runs")
    if cached:
        stats.cache_hits += 1
        return cached["total_count"]

    url = (
        f"https://api.github.com/repos/{repo_slug}/actions/workflows/{workflow_id}/runs?per_page=1"
    )
    try:
        data, _ = await github_request(
            client, url, stats, logger, token_state=token_state, pool=pool
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 451):
            return 0
        raise
    else:
        total = data.get("total_count", 0)
        cache.set(cache_key, "runs", {"total_count": total})
        logger.debug(
            f"Run count for {repo_slug} workflow {workflow_id}: {total}",
            extra={
                "event": "run_count_fetched",
                "repo": repo_slug,
                "workflow_id": workflow_id,
                "count": total,
            },
        )
        return total


def load_checkpoint() -> dict | None:
    if CHECKPOINT_FILE.exists():
        with CHECKPOINT_FILE.open("r") as f:
            return json.load(f)
    return None


def save_checkpoint(data: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("w") as f:
        json.dump(data, f, indent=2)


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_MINING_CONFIG
    if path.exists():
        with path.open("r") as f:
            return json.load(f)
    return {}


def get_current_criteria(config: dict) -> dict:
    filters = config.get("filters", {})
    return {
        "min_stars": filters.get("min_stars", 100),
        "min_commits": filters.get("min_repository_commits", 100),
        "min_runs": 100,
        "exclude_forks": filters.get("exclude_forks", True),
        "exclude_archived": filters.get("exclude_archived", True),
    }


def criteria_changed(current: dict, saved: dict) -> bool:
    return any(current.get(key) != saved.get(key) for key in CRITERIA_KEYS)


def _criteria_diff(current: dict, saved: dict) -> list[str]:
    diff = []
    for key in CRITERIA_KEYS:
        cur = current.get(key)
        old = saved.get(key)
        if cur != old:
            diff.append(f"  {key}: {old!r} → {cur!r}")
    return diff


def fmt_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def print_header():
    print()
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  Step 8: Online Dataset Enrichment{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    print()


def _format_token_table(pool: TokenPool) -> list[str]:
    lines = [f"  {BOLD}Token Status:{RESET}"]
    lines.append(f"  {'Owner':<25} {'Remaining':>10} {'Status':>10}")
    lines.append(f"  {'-' * 25} {'-' * 10} {'-' * 10}")
    for ts in pool.tokens:
        status = f"{RED}COOLDOWN{RESET}" if ts.cooldown else f"{GREEN}ACTIVE{RESET}"
        color = RED if ts.remaining < 100 else YELLOW if ts.remaining < 500 else GREEN
        lines.append(f"  {ts.owner:<25} {color}{ts.remaining:>10}{RESET} {status:>10}")
    return lines


def print_token_table(pool: TokenPool):
    for line in _format_token_table(pool):
        print(line)
    print()


def print_final_summary(stats: EnrichStats):
    print()
    print()
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  Enrichment Complete{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    print()
    start = datetime.fromisoformat(stats.started_at)
    end = datetime.fromisoformat(stats.finished_at)
    print(f"  {BOLD}Duration:{RESET}          {fmt_duration((end - start).total_seconds())}")
    print()
    print(f"  {BOLD}Repositories:{RESET}")
    print(f"    Total in input:       {stats.total_repos}")
    print(f"    Processed:            {stats.repos_processed}")
    print(f"    {GREEN}Enriched rows:        {stats.final_rows}{RESET}")
    print(f"    {RED}Excluded (fork):      {stats.repos_excluded_fork}{RESET}")
    print(f"    {RED}Excluded (archived):  {stats.repos_excluded_archived}{RESET}")
    print(f"    {YELLOW}Skipped (not found):  {stats.repos_skipped_not_found}{RESET}")
    print()
    print(f"  {BOLD}Workflows:{RESET}")
    print(f"    {GREEN}Enriched:             {stats.workflows_enriched}{RESET}")
    print(f"    {YELLOW}Excluded (no push/pr):{stats.workflows_excluded_no_push_pr}{RESET}")
    print(f"    {RED}Excluded (< 100 runs): {stats.workflows_excluded_low_runs}{RESET}")
    print()
    print(f"  {BOLD}API Performance:{RESET}")
    print(f"    Total calls:          {stats.api_calls}")
    print(f"    Errors:               {stats.api_errors}")
    print(f"    Cache hits:           {stats.cache_hits}")
    print(f"    Token rotations:      {stats.token_rotations}")
    print(f"    Freezes:              {stats.freezes}")
    print()


def _restore_stats(stats: EnrichStats, checkpoint: dict) -> None:
    stats.repos_processed = checkpoint.get("repos_processed", 0)
    stats.repos_excluded_fork = checkpoint.get("repos_excluded_fork", 0)
    stats.repos_excluded_archived = checkpoint.get("repos_excluded_archived", 0)
    stats.repos_skipped_not_found = checkpoint.get("repos_skipped_not_found", 0)
    stats.workflows_enriched = checkpoint.get("workflows_enriched", 0)
    stats.workflows_excluded_no_push_pr = checkpoint.get("workflows_excluded_no_push_pr", 0)
    stats.workflows_excluded_low_runs = checkpoint.get("workflows_excluded_low_runs", 0)
    stats.final_rows = checkpoint.get("final_rows", 0)
    stats.api_calls = checkpoint.get("api_calls", 0)
    stats.api_errors = checkpoint.get("api_errors", 0)
    stats.token_rotations = checkpoint.get("token_rotations", 0)
    stats.freezes = checkpoint.get("freezes", 0)
    stats.cache_hits = checkpoint.get("cache_hits", 0)


def _open_csv_writers(
    raw_path: Path,
    filtered_path: Path,
    fieldnames: list[str],
    append: bool,
) -> tuple[csv.DictWriter, csv.DictWriter, object, object]:
    if append:
        raw_file = raw_path.open("a", encoding="utf-8-sig", newline="")
        filtered_file = filtered_path.open("a", encoding="utf-8-sig", newline="")
    else:
        raw_file = raw_path.open("w", encoding="utf-8-sig", newline="")
        filtered_file = filtered_path.open("w", encoding="utf-8-sig", newline="")
    raw_writer = csv.DictWriter(raw_file, fieldnames=fieldnames)
    filtered_writer = csv.DictWriter(filtered_file, fieldnames=fieldnames)
    if not append:
        raw_writer.writeheader()
        filtered_writer.writeheader()
    return raw_writer, filtered_writer, raw_file, filtered_file


async def _enrich_single_workflow(
    http_client: httpx.AsyncClient,
    row: dict[str, str],
    wf: dict,
    triggers: list[str],
    repo_info: dict,
    repo_slug: str,
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState,
    raw_writer: csv.DictWriter,
    raw_file,
    filtered_writer: csv.DictWriter,
    filtered_file,
    pool: TokenPool | None = None,
) -> int:
    run_count = await fetch_run_count(
        http_client, repo_slug, wf["id"], cache, stats, logger, token_state, pool=pool
    )
    if run_count < 100:
        return 0

    enriched_row = dict(row)
    enriched_row["repo_is_fork"] = str(repo_info["is_fork"])
    enriched_row["repo_is_archived"] = str(repo_info["is_archived"])
    enriched_row["repo_default_branch"] = repo_info["default_branch"]
    enriched_row["workflow_triggers"] = json.dumps(triggers)
    enriched_row["workflow_run_count"] = str(run_count)

    raw_writer.writerow(enriched_row)
    filtered_writer.writerow(enriched_row)

    return 1


async def _acquire_token_async(
    pool: TokenPool, stats: EnrichStats, logger: logging.Logger
) -> TokenState:
    # Block until at least one token is usable. Unlike pool.get_token(),
    # this NEVER returns None — exhaustion leads to an async sleep until the
    # next reset, so callers don't silently skip repos on rate-limit burnout.
    while True:
        pool.reset_cooldowns()
        ts = pool.get_token()
        if ts is not None:
            return ts
        wait = pool.time_to_next_reset()
        wait = wait + 2 if wait > 0 else 15
        stats.freezes += 1
        logger.warning(
            "All tokens exhausted; sleeping %ds for reset",
            wait,
            extra={"event": "token_freeze", "wait_seconds": wait},
        )
        tqdm.write(f"  {RED}All tokens exhausted, sleeping {wait}s for reset...{RESET}")
        await asyncio.sleep(wait)


def _check_repo_exclusions(
    repo_slug: str,
    repo_info: dict,
    stats: EnrichStats,
    logger: logging.Logger,
    processed_repos: set[str],
    current_criteria: dict,
) -> bool:
    if repo_info is None or repo_info.get("not_found"):
        stats.repos_skipped_not_found += 1
        logger.warning(
            f"Skipping {repo_slug}: repo not found or API error",
            extra={"event": "repo_skipped", "repo": repo_slug, "reason": "not_found"},
        )
        _save_checkpoint_state(processed_repos, stats, current_criteria)
        return True

    if repo_info["is_fork"]:
        stats.repos_excluded_fork += 1
        logger.info(
            f"Excluded {repo_slug}: fork (EC1)",
            extra={
                "event": "repo_excluded",
                "repo": repo_slug,
                "criterion": "EC1",
                "reason": "fork",
            },
        )
        _save_checkpoint_state(processed_repos, stats, current_criteria)
        return True

    if repo_info["is_archived"]:
        stats.repos_excluded_archived += 1
        logger.info(
            f"Excluded {repo_slug}: archived (EC2)",
            extra={
                "event": "repo_excluded",
                "repo": repo_slug,
                "criterion": "EC2",
                "reason": "archived",
            },
        )
        _save_checkpoint_state(processed_repos, stats, current_criteria)
        return True

    return False


async def _fetch_workflows_map(
    http_client: httpx.AsyncClient,
    repo_slug: str,
    needed_paths: set[str],
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    token_state: TokenState,
    pool: TokenPool | None = None,
) -> dict:
    workflows = await fetch_workflows(
        http_client, repo_slug, cache, stats, logger, token_state, pool=pool
    )
    if workflows is None:
        workflows = []
    if needed_paths:
        workflows = [wf for wf in workflows if wf["path"] in needed_paths]
    return {wf["path"]: wf for wf in workflows}


async def _enrich_repo_workflows(
    http_client: httpx.AsyncClient,
    pool: TokenPool,
    repo_slug: str,
    repo_rows: list[dict],
    repo_info: dict,
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    raw_writer: csv.DictWriter,
    raw_file,
    filtered_writer: csv.DictWriter,
    filtered_file,
) -> int:
    needed_paths = {row["file_path"] for row in repo_rows}
    token_state = await _acquire_token_async(pool, stats, logger)
    workflow_map = await _fetch_workflows_map(
        http_client, repo_slug, needed_paths, cache, stats, logger, token_state, pool=pool
    )

    tasks = []
    for row in repo_rows:
        wf = workflow_map.get(row["file_path"])
        if wf is None:
            stats.workflows_excluded_no_push_pr += 1
            continue

        triggers = wf.get("triggers", [])
        if "push" not in triggers and "pull_request" not in triggers:
            stats.workflows_excluded_no_push_pr += 1
            continue

        run_token = await _acquire_token_async(pool, stats, logger)

        tasks.append(
            _enrich_single_workflow(
                http_client,
                row,
                wf,
                triggers,
                repo_info,
                repo_slug,
                cache,
                stats,
                logger,
                run_token,
                raw_writer,
                raw_file,
                filtered_writer,
                filtered_file,
                pool=pool,
            )
        )

    repo_enriched = 0
    if tasks:
        results = await asyncio.gather(*tasks)
        for enriched in results:
            if enriched:
                repo_enriched += 1
            else:
                stats.workflows_excluded_low_runs += 1

    stats.repos_processed += 1
    stats.workflows_enriched += repo_enriched
    stats.final_rows += repo_enriched
    logger.info(
        f"Completed {repo_slug}: {repo_enriched}/{len(repo_rows)} workflows enriched",
        extra={
            "event": "repo_complete",
            "repo": repo_slug,
            "enriched": repo_enriched,
            "total": len(repo_rows),
        },
    )
    return repo_enriched


async def _process_repo(
    http_client: httpx.AsyncClient,
    pool: TokenPool,
    repo_slug: str,
    repo_rows: list[dict],
    cache: GitHubCache,
    stats: EnrichStats,
    logger: logging.Logger,
    current_criteria: dict,
    processed_repos: set[str],
    raw_writer: csv.DictWriter,
    raw_file,
    filtered_writer: csv.DictWriter,
    filtered_file,
) -> bool:
    cached_info = cache.get(repo_slug, "repos")
    if cached_info:
        stats.cache_hits += 1
        repo_info = cached_info
    else:
        ts = await _acquire_token_async(pool, stats, logger)
        repo_info = await fetch_repo_info(
            http_client, repo_slug, cache, stats, logger, token_state=ts, pool=pool
        )

    processed_repos.add(repo_slug)

    if _check_repo_exclusions(
        repo_slug, repo_info, stats, logger, processed_repos, current_criteria
    ):
        return True

    await _enrich_repo_workflows(
        http_client,
        pool,
        repo_slug,
        repo_rows,
        repo_info,
        cache,
        stats,
        logger,
        raw_writer,
        raw_file,
        filtered_writer,
        filtered_file,
    )
    return True


def _maybe_checkpoint(
    stats: EnrichStats,
    processed_repos: set[str],
    current_criteria: dict,
    checkpoint_every: int,
    pool: TokenPool,
) -> None:
    if stats.repos_processed % checkpoint_every == 0:
        _save_checkpoint_state(processed_repos, stats, current_criteria)
        elapsed = time.time() - datetime.fromisoformat(stats.started_at).timestamp()
        rate = stats.repos_processed / (elapsed / 3600) if elapsed > 0 else 0
        tqdm.write(
            f"  {CYAN}Checkpoint #{stats.repos_processed // checkpoint_every}: "
            f"{stats.repos_processed}/{stats.total_repos} repos, "
            f"{stats.final_rows} rows, "
            f"{stats.api_calls} API calls, "
            f"{rate:.0f} repos/hr{RESET}"
        )
        tqdm.write("")
        for line in _format_token_table(pool):
            tqdm.write(line)
        tqdm.write("")


def _csv_has_data_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        return next(reader, None) is not None


def _rebuild_csvs_from_cache(
    repos: dict[str, list[dict[str, str]]],
    processed_repos: set[str],
    cache: GitHubCache,
    raw_path: Path,
    output_path: Path,
    fieldnames: list[str],
    current_criteria: dict,
    stats: EnrichStats,
) -> int:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        raw_path.open("w", encoding="utf-8-sig", newline="") as raw_file,
        output_path.open("w", encoding="utf-8-sig", newline="") as filtered_file,
    ):
        raw_writer = csv.DictWriter(raw_file, fieldnames=fieldnames)
        filtered_writer = csv.DictWriter(filtered_file, fieldnames=fieldnames)
        raw_writer.writeheader()
        filtered_writer.writeheader()

        count = 0
        # sorted() makes the rebuilt CSV deterministic regardless of the
        # order in which the underlying set was populated.
        for repo_slug in sorted(processed_repos):
            repo_rows = repos.get(repo_slug, [])
            if not repo_rows:
                continue
            rows_added = _rebuild_repo_from_cache(
                repo_slug,
                repo_rows,
                cache,
                current_criteria,
                raw_writer,
                filtered_writer,
                stats,
            )
            count += rows_added
            stats.repos_processed += 1

    return count


def _rebuild_repo_from_cache(
    repo_slug: str,
    repo_rows: list[dict[str, str]],
    cache: GitHubCache,
    criteria: dict,
    raw_writer: csv.DictWriter,
    filtered_writer: csv.DictWriter,
    stats: EnrichStats,
) -> int:
    repo_info = cache.get(repo_slug, "repos")
    if repo_info is None:
        return 0
    if repo_info.get("not_found"):
        return 0

    if repo_info["is_fork"] and criteria.get("exclude_forks"):
        stats.repos_excluded_fork += 1
        return 0
    if repo_info["is_archived"] and criteria.get("exclude_archived"):
        stats.repos_excluded_archived += 1
        return 0

    workflows_cached = cache.get(repo_slug, "workflows")
    if workflows_cached is None:
        return 0
    workflow_map = {wf["path"]: wf for wf in workflows_cached}

    enriched = 0
    for row in repo_rows:
        wf = workflow_map.get(row["file_path"])
        if wf is None:
            stats.workflows_excluded_no_push_pr += 1
            continue
        triggers = wf.get("triggers", [])
        if "push" not in triggers and "pull_request" not in triggers:
            stats.workflows_excluded_no_push_pr += 1
            continue

        cache_key = f"{repo_slug}_{wf['id']}"
        run_cached = cache.get(cache_key, "runs")
        if run_cached is None:
            continue
        if run_cached["total_count"] < 100:
            stats.workflows_excluded_low_runs += 1
            continue

        enriched_row = dict(row)
        enriched_row["repo_is_fork"] = str(repo_info["is_fork"])
        enriched_row["repo_is_archived"] = str(repo_info["is_archived"])
        enriched_row["repo_default_branch"] = repo_info["default_branch"]
        enriched_row["workflow_triggers"] = json.dumps(triggers)
        enriched_row["workflow_run_count"] = str(run_cached["total_count"])

        raw_writer.writerow(enriched_row)
        filtered_writer.writerow(enriched_row)
        enriched += 1

    stats.workflows_enriched += enriched
    stats.final_rows += enriched
    return enriched


def _load_enriched_repos(raw_path: Path) -> set[str]:
    """Return repo slugs already present in the raw enriched CSV.

    Used to skip repos that were processed in a previous completed run
    (when the checkpoint file was already deleted on success).
    """
    if not raw_path.exists():
        return set()
    repos: set[str] = set()
    with raw_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row.get("repository", "")
            if repo:
                repos.add(repo)
    return repos


def _load_or_init_checkpoint(
    current_criteria: dict,
    stats: EnrichStats,
    logger: logging.Logger,
    raw_csv: Path,
    output_path: Path,
    fieldnames: list[str],
    repos: dict[str, list[dict[str, str]]],
    cache: GitHubCache,
) -> set[str]:
    from dataset_utils.integrity import PipelineIntegrityError

    processed_repos: set[str] = set()
    checkpoint = load_checkpoint()

    if checkpoint:
        processed_repos = set(checkpoint.get("processed_repos", []))
        saved_criteria = checkpoint.get("criteria_snapshot", {})

        if criteria_changed(current_criteria, saved_criteria):
            diff_lines = "\n".join(_criteria_diff(current_criteria, saved_criteria))
            msg = (
                "Criteria changed since last checkpoint. Resuming with different "
                "criteria can silently drop or miss repositories (e.g. relaxing "
                "exclude_forks would not re-fetch repos already excluded as forks).\n"
                f"Differences:\n{diff_lines}\n\n"
                "To re-mine with the new criteria, manually wipe progress:\n"
                f"  rm {CHECKPOINT_FILE} {RAW_CSV} {output_path}\n"
                "If you also want to discard cached HTTP responses (forces re-fetch "
                "of repo metadata), additionally:\n"
                f"  rm -rf {HTTP_CACHE_DIR}\n"
                "Then re-run Step 8."
            )
            logger.error(
                "Criteria changed since checkpoint; aborting to avoid silent data loss",
                extra={
                    "event": "criteria_changed_abort",
                    "saved": saved_criteria,
                    "current": current_criteria,
                },
            )
            raise PipelineIntegrityError(msg)

        _restore_stats(stats, checkpoint)
        csvs_valid = _csv_has_data_rows(raw_csv)
        if csvs_valid:
            logger.info(
                "Resumed from checkpoint with valid CSVs: %d repos already processed",
                len(processed_repos),
                extra={"event": "resume", "processed": len(processed_repos)},
            )
        else:
            msg = f"  {YELLOW}Rebuilding CSVs from cache for {len(processed_repos)} repos...{RESET}"
            print(msg)
            count = _rebuild_csvs_from_cache(
                repos,
                processed_repos,
                cache,
                raw_csv,
                output_path,
                fieldnames,
                current_criteria,
                stats,
            )
            if count > 0:
                print(f"  {GREEN}Rebuilt CSVs: {count} rows written{RESET}")
                logger.info(
                    "Rebuilt CSVs from cache: %d rows for %d repos",
                    count,
                    len(processed_repos),
                    extra={
                        "event": "csv_rebuilt",
                        "rows": count,
                        "repos": len(processed_repos),
                    },
                )
            else:
                msg = (
                    f"  {RED}Cache incomplete (no run counts). "
                    f"Reprocessing repos (cache still speeds up "
                    f"repo/workflow lookups){RESET}"
                )
                print(msg)
                processed_repos.clear()
                for key in [
                    "repos_processed",
                    "repos_excluded_fork",
                    "repos_excluded_archived",
                    "repos_skipped_not_found",
                    "workflows_enriched",
                    "workflows_excluded_no_push_pr",
                    "workflows_excluded_low_runs",
                    "final_rows",
                ]:
                    setattr(stats, key, 0)

    return processed_repos


def _read_input_csv(input_path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(_MSG_NO_HEADER)

        base_fieldnames = list(reader.fieldnames)
        enrichment_fields = [
            "repo_is_fork",
            "repo_is_archived",
            "repo_default_branch",
            "workflow_triggers",
            "workflow_run_count",
        ]
        fieldnames = base_fieldnames + enrichment_fields

        repos: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            repo = row["repository"]
            if repo not in repos:
                repos[repo] = []
            repos[repo].append(row)

    return fieldnames, repos


def _setup_interrupt_handler(
    processed_repos: set[str],
    stats: EnrichStats,
    current_criteria: dict,
    raw_file,
    filtered_file,
):
    # Use a threading.Event so the async loop can exit cleanly instead of
    # calling sys.exit(0) from a signal handler, which raises SystemExit
    # mid-coroutine and produces noisy tracebacks in Python 3.12+.
    import threading

    _stop_event = threading.Event()

    def handle_interrupt(signum, frame):
        if _stop_event.is_set():
            return
        _stop_event.set()
        print(f"\n\n  {YELLOW}Interrupt received, saving checkpoint...{RESET}")
        _save_checkpoint_state(processed_repos, stats, current_criteria)
        for f in (raw_file, filtered_file):
            if f:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
        print(f"  {GREEN}Checkpoint saved. Exiting safely.{RESET}")
        os._exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)


def _log_final_results(stats: EnrichStats, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("Step 8: Online dataset enrichment completed")
    logger.info(
        "Duration: %s",
        datetime.fromisoformat(stats.finished_at) - datetime.fromisoformat(stats.started_at),
    )
    logger.info(f"Total repos: {stats.total_repos}")
    logger.info(f"Repos processed: {stats.repos_processed}")
    logger.info(f"Repos excluded (fork): {stats.repos_excluded_fork}")
    logger.info(f"Repos excluded (archived): {stats.repos_excluded_archived}")
    logger.info(f"Repos skipped (not found): {stats.repos_skipped_not_found}")
    logger.info(f"Workflows enriched: {stats.workflows_enriched}")
    logger.info(f"Workflows excluded (no push/pr): {stats.workflows_excluded_no_push_pr}")
    logger.info(f"Workflows excluded (< 100 runs): {stats.workflows_excluded_low_runs}")
    logger.info(f"Final rows: {stats.final_rows}")
    logger.info(f"API calls: {stats.api_calls}")
    logger.info(f"API errors: {stats.api_errors}")
    logger.info(f"Token rotations: {stats.token_rotations}")
    logger.info(f"Freezes: {stats.freezes}")
    logger.info(f"Cache hits: {stats.cache_hits}")
    logger.info("=" * 60)


def _load_tokens(config: dict) -> list[dict]:
    # Reject placeholder tokens from the example config — they trip a 401
    # on the first API call. Treat them as if no tokens were configured so
    # the env-var fallback can take over.
    raw = config.get("github", {}).get("tokens", [])
    tokens_config = [t for t in raw if t.get("token") and not t["token"].startswith("REPLACE_")]
    if not tokens_config:
        env_token = os.environ.get("GITHUB_TOKEN")
        if env_token:
            tokens_config = [{"owner": "env", "token": env_token}]
        else:
            raise ValueError(_MSG_NO_TOKENS)
    return tokens_config


async def _enrich_dataset_async(
    input_path: Path, output_path: Path, verbose: bool = False, limit: int | None = None
) -> EnrichStats:
    logger = setup_logging(verbose)
    stats = EnrichStats()
    stats.started_at = datetime.now().isoformat()

    print_header()

    config = load_config()
    tokens_config = _load_tokens(config)
    safety_threshold = config.get("rate_limit", {}).get("safety_threshold_remaining", 15)
    checkpoint_every = config.get("mining", {}).get("checkpoint_every_repos", 25)
    repo_concurrency = config.get("mining", {}).get("repo_concurrency", max(1, len(tokens_config)))
    current_criteria = get_current_criteria(config)

    pool = TokenPool(tokens_config, safety_threshold)
    cache = GitHubCache(HTTP_CACHE_DIR, CORPUS_DIR)
    http_client = httpx.AsyncClient(
        headers={"Accept": "application/vnd.github.v3+json"},
        timeout=30,
        http2=True,
        follow_redirects=True,
    )

    print(
        f"  {BOLD}Tokens:{RESET} {len(pool.tokens)} configured,"
        f" safety threshold: {safety_threshold}"
    )
    print(f"  {BOLD}Checkpoint every:{RESET} {checkpoint_every} repos")
    print(f"  {BOLD}Repo concurrency:{RESET} {repo_concurrency}")
    print_token_table(pool)

    fieldnames, repos = _read_input_csv(input_path)

    if limit is not None and limit > 0:
        truncated = dict(list(repos.items())[:limit])
        print(
            f"  {YELLOW}{BOLD}SMOKE-TEST MODE:{RESET} processing first "
            f"{len(truncated)} of {len(repos)} repos"
        )
        repos = truncated

    processed_repos = _load_or_init_checkpoint(
        current_criteria, stats, logger, RAW_CSV, output_path, fieldnames, repos, cache
    )

    # Supplement with repos already present in the raw output CSV.
    # Handles the common case where a previous run completed successfully
    # (checkpoint was deleted) and the input now contains additional repos.
    already_in_raw = _load_enriched_repos(RAW_CSV)
    new_from_raw = already_in_raw - processed_repos
    if new_from_raw:
        logger.info(
            "Found %d repos already in raw CSV but not in checkpoint — skipping them",
            len(new_from_raw),
        )
        print(
            f"  {CYAN}Skipping {len(new_from_raw)} repos already in raw CSV"
            f" (previous completed run){RESET}"
        )
        processed_repos |= new_from_raw

    stats.total_repos = len(repos)
    remaining = stats.total_repos - len(processed_repos)

    print(f"  {BOLD}Input repos:{RESET} {stats.total_repos}")
    print(f"  {BOLD}Already processed:{RESET} {len(processed_repos)}")
    print(f"  {BOLD}Remaining:{RESET} {remaining}")
    print()

    # Append to RAW_CSV when it exists — _load_enriched_repos ensures
    # already-processed repos are skipped. Clearing RAW_CSV on resume would
    # destroy data from prior runs.
    append_mode = RAW_CSV.exists()
    raw_writer, filtered_writer, raw_file, filtered_file = _open_csv_writers(
        RAW_CSV, output_path, fieldnames, append_mode
    )

    _setup_interrupt_handler(processed_repos, stats, current_criteria, raw_file, filtered_file)

    last_progress_time = time.time()
    progress_interval = 5.0

    repos_to_process = {k: v for k, v in repos.items() if k not in processed_repos}
    repo_items = list(repos_to_process.items())
    semaphore = asyncio.Semaphore(repo_concurrency)

    async def _process_repo_limited(repo_slug: str, repo_rows: list[dict]) -> bool:
        async with semaphore:
            return await _process_repo(
                http_client,
                pool,
                repo_slug,
                repo_rows,
                cache,
                stats,
                logger,
                current_criteria,
                processed_repos,
                raw_writer,
                raw_file,
                filtered_writer,
                filtered_file,
            )

    with tqdm(
        total=len(repos),
        initial=stats.repos_processed,
        desc="Enriching repos",
        unit="repo",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        colour="green",
    ) as pbar:
        iterator = iter(repo_items)
        in_flight: set[asyncio.Task] = set()

        for _ in range(repo_concurrency):
            nxt = next(iterator, None)
            if nxt is None:
                break
            repo_slug, repo_rows = nxt
            in_flight.add(asyncio.create_task(_process_repo_limited(repo_slug, repo_rows)))

        while in_flight:
            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
                pbar.update(1)
            for _ in range(len(done)):
                nxt = next(iterator, None)
                if nxt is None:
                    break
                repo_slug, repo_rows = nxt
                in_flight.add(asyncio.create_task(_process_repo_limited(repo_slug, repo_rows)))

                now = time.time()
                if now - last_progress_time >= progress_interval:
                    pbar.set_postfix(
                        rows=stats.final_rows,
                        api=stats.api_calls,
                        cache=stats.cache_hits,
                        refresh=True,
                    )
                    last_progress_time = now

                if stats.final_rows % 100 == 0:
                    raw_file.flush()
                    filtered_file.flush()
                _maybe_checkpoint(stats, processed_repos, current_criteria, checkpoint_every, pool)

    await http_client.aclose()
    if raw_file:
        raw_file.close()
    if filtered_file:
        filtered_file.close()

    stats.finished_at = datetime.now().isoformat()
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    # Post-process: stable (repository, file_path) ordering so CSV hashes
    # match across runs regardless of async task completion order.
    _normalize_output(RAW_CSV, fieldnames)
    _normalize_output(output_path, fieldnames)

    _write_step8_report(stats, output_path)
    _package_corpus()

    _log_final_results(stats, logger)

    print_final_summary(stats)

    return stats


def _normalize_output(path: Path, fieldnames: list[str]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda r: (r.get("repository", ""), r.get("file_path", "")))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_step8_report(stats: EnrichStats, output_path: Path) -> None:
    from dataclasses import asdict

    from dataset_utils.hashing import compute_file_hash

    report = {
        **asdict(stats),
        "input_sha256": _STEP8_INPUT_SHA,
        "output_file": str(output_path),
        "output_sha256": compute_file_hash(output_path) if output_path.exists() else None,
        "raw_output_file": str(RAW_CSV),
        "raw_output_sha256": compute_file_hash(RAW_CSV) if RAW_CSV.exists() else None,
        "corpus_dir": str(CORPUS_DIR),
        "corpus_file_count": (sum(1 for _ in CORPUS_DIR.iterdir()) if CORPUS_DIR.is_dir() else 0),
    }
    report_path = output_path.parent / "step8_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)


def _package_corpus(verbose: bool = True) -> None:
    """Compress `data/downloaded_workflows/` → sibling `.tar.xz` + `.sha256.txt` sidecar."""
    from dataset_utils.bootstrap import package_dir

    if not CORPUS_DIR.is_dir():
        if verbose:
            print(f"  {YELLOW}No corpus dir at {CORPUS_DIR}; skipping packaging.{RESET}")
        return
    archive, sidecar = package_dir(CORPUS_DIR, replace=True)
    if verbose:
        print(f"  {GREEN}Packaged corpus:{RESET} {archive.name} + {sidecar.name}")


def enrich_dataset(
    input_path: Path,
    output_path: Path,
    verbose: bool = False,
    limit: int | None = None,
) -> EnrichStats:
    return asyncio.run(_enrich_dataset_async(input_path, output_path, verbose=verbose, limit=limit))


def _save_checkpoint_state(processed_repos: set, stats: EnrichStats, criteria: dict):
    save_checkpoint(
        {
            "processed_repos": list(processed_repos),
            "criteria_snapshot": criteria,
            "repos_processed": stats.repos_processed,
            "repos_excluded_fork": stats.repos_excluded_fork,
            "repos_excluded_archived": stats.repos_excluded_archived,
            "repos_skipped_not_found": stats.repos_skipped_not_found,
            "workflows_enriched": stats.workflows_enriched,
            "workflows_excluded_no_push_pr": stats.workflows_excluded_no_push_pr,
            "workflows_excluded_low_runs": stats.workflows_excluded_low_runs,
            "final_rows": stats.final_rows,
            "api_calls": stats.api_calls,
            "api_errors": stats.api_errors,
            "token_rotations": stats.token_rotations,
            "freezes": stats.freezes,
            "cache_hits": stats.cache_hits,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich filtered dataset with GitHub API data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input filtered dataset CSV file path. (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output enriched dataset CSV file path. (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=DEFAULT_PRIOR_REPORT,
        help=f"Step 7 report path used for SHA-chain check. (default: {DEFAULT_PRIOR_REPORT})",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        default=False,
        help="Skip the upstream SHA-chain check (use only when running Step 8 standalone).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging to console.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most N repositories from the input (smoke-test mode). "
            "Skips checkpoint/resume logic and writes outputs to data/.smoketest/ "
            "to avoid polluting the primary Step 8 outputs."
        ),
    )
    return parser.parse_args()


_STEP8_INPUT_SHA: str | None = None


def _activate_smoketest_paths() -> Path:
    """Redirect all module-level output paths to data/.smoketest/.

    Mutates module globals so the rest of the script writes the sandbox
    instead of the primary Step 8 outputs. Returns the smoketest output CSV.
    """
    global \
        CHECKPOINT_FILE, \
        HTTP_CACHE_DIR, \
        CORPUS_DIR, \
        LOG_DIR, \
        LOG_FILE, \
        LOG_JSON_FILE, \
        RAW_CSV, \
        FILTERED_CSV
    sandbox = DATA_DIR / ".smoketest"
    sandbox.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE = sandbox / ".checkpoint_enrich.json"
    HTTP_CACHE_DIR = sandbox / ".http_cache"
    CORPUS_DIR = sandbox / "downloaded_workflows"
    LOG_DIR = sandbox / ".logs"
    LOG_FILE = LOG_DIR / "enrichment.log"
    LOG_JSON_FILE = LOG_DIR / "enrichment.jsonl"
    RAW_CSV = sandbox / "8_dataset_enriched_raw.csv"
    FILTERED_CSV = sandbox / "8_dataset_enriched.csv"
    return FILTERED_CSV


def main() -> None:
    args = parse_args()

    from dataset_utils.hashing import compute_file_hash
    from dataset_utils.integrity import PipelineIntegrityError, assert_input_matches_report

    smoketest = args.limit is not None and args.limit > 0
    if smoketest:
        sandbox_output = _activate_smoketest_paths()
        if args.output == DEFAULT_OUTPUT_CSV:
            args.output = sandbox_output
        args.skip_integrity_check = True

    global _STEP8_INPUT_SHA
    if args.skip_integrity_check:
        _STEP8_INPUT_SHA = compute_file_hash(args.input) if args.input.exists() else None
    else:
        _STEP8_INPUT_SHA = assert_input_matches_report(
            input_path=args.input,
            prior_report_path=args.prior_report,
            label="step 8 input",
        )

    try:
        enrich_dataset(args.input, args.output, verbose=args.verbose, limit=args.limit)
    except KeyboardInterrupt:
        sys.exit(0)
    except PipelineIntegrityError as e:
        print(f"\n{RED}{BOLD}Step 8 aborted:{RESET} {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
