import io
import re
from os.path import basename

from ruamel.yaml import YAML


def parse_yaml(yaml_str: str | None) -> dict | None:
    if yaml_str is None:
        return None
    yaml = YAML()
    try:
        content = yaml.load(io.StringIO(yaml_str))
        return content
    except Exception as e:
        raise ValueError(f"Unable to parse yaml file: {e}") from e


def print_smells(smells: set) -> None:
    print(f"We have found {len(smells)} smells")
    ordered_smells = list(smells)
    ordered_smells.sort(key=lambda x: int(x.split(".")[0]))
    for s in ordered_smells:
        print("\t- " + s)


SMELL_NAME_BY_ID = {
    1: "ForkSafety",
    2: "ForkSafety",
    3: "RunnerVersioning",
    4: "ConcurrencyControl",
    5: "ConcurrencyControl",
    6: "Permissions",
    7: "ForkSafety",
    8: "ActionVersioning",
    9: "StepDesign",
    10: "ErrorHandling",
    11: "ForkSafety",
    12: "Documentation",
    13: "Readability",
    14: "Formatting",
    15: "Permissions",
    16: "Efficiency",
    17: "ConcurrencyControl",
    18: "DependencyManagement",
    19: "Compatibility",
    20: "Compatibility",
    21: "Caching",
    22: "DeploymentSafety",
}


# Tool-internal ID → (Khatami paper ID, PascalCase identifier).
# Source: Khatami et al. SCAM 2024, Table III. The CSV description column
# emits the PascalCase identifier so that downstream consumers have a
# single stable key per smell (no whitespace, no apostrophes, no prefix/
# suffix drift). The Khatami paper ordering differs from the tool-internal
# ordering; paper_id is the authoritative ID.
KHATAMI_BY_TOOL_ID: dict[int, tuple[int, str]] = {
    1:  (9,  "MissingForkGuardSchedule"),
    2:  (4,  "MissingForkGuardPR"),
    3:  (15, "MutableRunnerVersion"),
    4:  (7,  "MissingConcurrencyBranch"),
    5:  (6,  "MissingConcurrencyPR"),
    6:  (1,  "MissingPermExternalActions"),
    7:  (11, "MissingUploadCondition"),
    8:  (2,  "UnpinnedActionVersion"),
    9:  (18, "MultiCommandStep"),
    10: (5,  "MissingJobTimeout"),
    11: (10, "UnsafeArtifactUploadFork"),
    12: (21, "AddCommentsToWorkflows"),
    13: (16, "NameRunSteps"),
    14: (14, "CorrectIndentation"),
    15: (3,  "MissingTokenPermissions"),
    16: (8,  "MissingPathFilters"),
    17: (13, "MissingConcurrencyGlobal"),
    18: (20, "UnpinnedDependencies"),
    19: (19, "SingleOSExecution"),
    20: (22, "SingleLangVersionCI"),
    21: (17, "SuboptimalCacheUsage"),
    22: (12, "UnsafeDeployOnFork"),
}


def _smell_id(smell: str) -> int | None:
    match = re.match(r"^\s*(\d+)\.", smell)
    if match is None:
        return None
    return int(match.group(1))


def _extract_line(smell: str) -> int | None:
    line_patterns = [
        r"line\s*:?\s*(-?\d+)",
        r"lines\s+(\d+)\s*:\s*\d+",
        r"job\s+at\s+line\s*:?\s*(-?\d+)",
    ]
    for pattern in line_patterns:
        match = re.search(pattern, smell, flags=re.IGNORECASE)
        if match is not None:
            value = int(match.group(1))
            if value > 0:
                return value
            return None
    return None


def _strip_line_suffix(smell: str) -> str:
    description = re.sub(
        r"\s*\((line|lines|job at line).*$", "", smell, flags=re.IGNORECASE
    )
    description = re.sub(r"^\s*\d+\.\s*", "", description)
    return description.strip()


def _indent_level(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _find_block_start(lines: list[str], line_index: int) -> int:
    current = lines[line_index]
    current_indent = _indent_level(current)

    if current.strip().endswith(":"):
        return line_index

    for idx in range(line_index - 1, -1, -1):
        stripped = lines[idx].strip()
        if stripped == "":
            continue
        if stripped.endswith(":") and _indent_level(lines[idx]) < current_indent:
            return idx
    return line_index


def _code_for_line(file_content: str, line: int | None) -> str:
    if line is None or line <= 0:
        return ""
    lines = file_content.split("\n")
    if line > len(lines):
        return ""

    block_start = _find_block_start(lines, line - 1)
    base_indent = _indent_level(lines[block_start])
    block_end = block_start + 1

    for idx in range(block_start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped == "":
            block_end = idx + 1
            continue
        if _indent_level(lines[idx]) <= base_indent:
            break
        block_end = idx + 1

    block_lines = lines[block_start:block_end]
    while len(block_lines) > 0 and block_lines[-1].strip() == "":
        block_lines.pop()
    return "\n".join(block_lines)


def _find_line_by_regex(file_content: str, pattern: str) -> int | None:
    lines = file_content.split("\n")
    regex = re.compile(pattern)
    for idx, line in enumerate(lines, start=1):
        if regex.search(line):
            return idx
    return None


def _first_job_line(file_content: str) -> int | None:
    lines = file_content.split("\n")
    jobs_line = None
    jobs_indent = 0
    for idx, line in enumerate(lines, start=1):
        if line.strip() == "jobs:":
            jobs_line = idx
            jobs_indent = _indent_level(line)
            break
    if jobs_line is None:
        return None

    for idx in range(jobs_line + 1, len(lines) + 1):
        line = lines[idx - 1]
        if line.strip() == "":
            continue
        if _indent_level(line) > jobs_indent and line.strip().endswith(":"):
            return idx
        if _indent_level(line) <= jobs_indent:
            break
    return jobs_line


def _line_from_job_name(file_content: str, job_name: str) -> int | None:
    escaped = re.escape(job_name.strip())
    return _find_line_by_regex(file_content, rf"^\s*{escaped}\s*:\s*$")


def _line_from_run_smell(file_content: str) -> int | None:
    lines = file_content.split("\n")
    for idx, line in enumerate(lines, start=1):
        if "run:" not in line:
            continue
        if "&&" in line:
            return idx
        current_indent = _indent_level(line)
        for next_idx in range(idx + 1, len(lines) + 1):
            nxt = lines[next_idx - 1]
            if nxt.strip() == "":
                continue
            if _indent_level(nxt) > current_indent:
                return idx
            break
    return None


def _infer_line(
    smell: str, smell_id: int | None, file_content: str
) -> int | None:
    # Job-scoped smells may encode a job name rather than a numeric line.
    job_match = re.search(r"\(job\s*:\s*([^)]+)\)", smell, flags=re.IGNORECASE)
    if job_match is not None:
        line = _line_from_job_name(file_content, job_match.group(1))
        if line is not None:
            return line

    if smell_id == 9:
        line = _line_from_run_smell(file_content)
        if line is not None:
            return line

    if smell_id == 12:
        return 1

    if smell_id in {19, 20, 22}:
        line = _find_line_by_regex(file_content, r"^\s*jobs\s*:\s*$")
        if line is not None:
            return line
        return _first_job_line(file_content)

    if smell_id == 16:
        line = _find_line_by_regex(file_content, r"^\s*on\s*:\s*$")
        if line is not None:
            return line

    return None


def smells_to_csv_rows(
    workflow_name: str, file_content: str, smells: set[str]
) -> list[list[str]]:
    rows: list[list[str]] = []
    ordered_smells = list(smells)
    ordered_smells.sort()
    for smell in ordered_smells:
        smell_id = _smell_id(smell)
        smell_name = SMELL_NAME_BY_ID.get(smell_id, "General")
        # Emit Khatami Table III PascalCase identifier as description.
        # Multiple tool-raw variants for the same ID (e.g., ID 22 has both
        # "Avoid deploying jobs on forks" and "Avoid deploying from forks")
        # collapse to one PascalCase key — downstream dedups.
        khatami = KHATAMI_BY_TOOL_ID.get(smell_id)
        if khatami is None:
            # Fall back to stripped raw for unknown IDs (no data loss, still debuggable).
            description = _strip_line_suffix(smell)
        else:
            description = khatami[1]
        line = _extract_line(smell)
        if line is None:
            line = _infer_line(smell, smell_id, file_content)
        code = _code_for_line(file_content, line)
        rows.append(
            [
                basename(workflow_name),
                smell_name,
                description,
                "" if line is None else str(line),
                code,
            ]
        )
    return rows


def styling_to_csv_rows(
    workflow_name: str, file_content: str, styling: list
) -> list[list[str]]:
    rows: list[list[str]] = []
    lines = file_content.split("\n")
    for problem in styling:
        line = getattr(problem, "line", None)
        code = lines[line - 1] if line is not None and 0 < line <= len(lines) else ""
        rows.append(
            [
                basename(workflow_name),
                "Formatting",
                str(problem),
                "" if line is None else str(line),
                code,
            ]
        )
    return rows


def _escape_csv_string(value: str) -> str:
    return value.replace('"', '""')


def _write_csv_rows(csv_file, rows: list[list[str]]) -> None:
    for row in rows:
        file_name = _escape_csv_string(str(row[0]))
        smell = _escape_csv_string(str(row[1]))
        description = _escape_csv_string(str(row[2]))
        line = _escape_csv_string(str(row[3]))
        code = _escape_csv_string(str(row[4]).rstrip("\r\n"))
        csv_file.write(f'{file_name},{smell},"{description}",{line},"{code}"\n')


def write_csv_header(output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_file.write("file,smell,description,line,code\n")


def append_smells_csv(output_path: str, rows: list[list[str]]) -> None:
    if not rows:
        return
    with open(output_path, "a", newline="", encoding="utf-8") as csv_file:
        _write_csv_rows(csv_file, rows)


def write_smells_csv(output_path: str, rows: list[list[str]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_file.write("file,smell,description,line,code\n")
        _write_csv_rows(csv_file, rows)


def fill_dict(mods: dict[int, int]) -> (list[int], list[int]):
    if len(mods.keys()) == 0:
        return [], []
    time_stamps = list(range(1, list(mods.keys())[-1]))
    counts = []
    last = 0
    for el in time_stamps:
        try:
            last = mods[el] + last
            counts.append(last)
        except KeyError:
            counts.append(last)
    return counts, time_stamps
