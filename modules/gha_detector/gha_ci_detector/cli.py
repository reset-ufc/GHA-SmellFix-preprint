import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from gha_ci_detector import __app_name__, __version__, util
from gha_ci_detector.Runner import Runner
from gha_ci_detector.Workflow import Workflow

app = typer.Typer()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FileAnalysisResult:
    file_path: str
    rows: list[list[str]]
    smell_count: int
    category_counts: Counter
    attempts: int
    error: str | None = None
    non_retriable: bool = False
    report_lines: list[str] | None = None


class NonRetriableWorkflowError(ValueError):
    pass


def _resolve_output_path(path: str) -> str:
    output_path = Path(path)
    if output_path.is_absolute():
        return str(output_path)
    return str(PROJECT_ROOT / output_path)


# def main():
#     app(prog_name=__app_name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show application version and exit",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    return


def analyze_and_report_workflow(workflow: Workflow, report: bool = True) -> set[str]:
    runner = Runner(workflow)
    smells = runner.run_all()
    if report:
        util.print_smells(smells)
        if len(workflow.styling) > 0:
            typer.echo("The following styling errors were found:")
            for s in workflow.styling:
                typer.echo(s)
    return smells


def _build_workflow_report_lines(workflow: Workflow, smells: set[str]) -> list[str]:
    report_lines: list[str] = []
    report_lines.append(
        f"[{os.path.basename(workflow.name)}] Found {len(smells)} smells"
    )
    ordered_smells = sorted(smells, key=lambda x: int(x.split(".")[0]))
    for smell in ordered_smells:
        report_lines.append(f"  - {smell}")
    if len(workflow.styling) > 0:
        report_lines.append("  - Styling errors:")
        for problem in workflow.styling:
            report_lines.append(f"    - {problem}")
    return report_lines


def _validate_workflow_contract(workflow: Workflow) -> None:
    if not isinstance(workflow.yaml, dict):
        raise NonRetriableWorkflowError("workflow root must be a YAML mapping")

    jobs = workflow.yaml.get("jobs")
    if not isinstance(jobs, dict):
        raise NonRetriableWorkflowError("workflow 'jobs' must be a mapping")

    for job_name, job_yaml in jobs.items():
        if not isinstance(job_yaml, dict):
            raise NonRetriableWorkflowError(f"job '{job_name}' must be a mapping")

        if "steps" in job_yaml:
            if not isinstance(job_yaml["steps"], list):
                raise NonRetriableWorkflowError(
                    f"job '{job_name}' has non-list 'steps'"
                )

            for step_index, step_yaml in enumerate(job_yaml["steps"], start=1):
                if not isinstance(step_yaml, dict):
                    raise NonRetriableWorkflowError(
                        f"job '{job_name}' step #{step_index} must be a mapping"
                    )

                if "run" in step_yaml and not isinstance(step_yaml["run"], str):
                    step_yaml["run"] = str(step_yaml["run"])

                if "uses" in step_yaml and not isinstance(step_yaml["uses"], str):
                    raise NonRetriableWorkflowError(
                        f"job '{job_name}' step #{step_index} has non-string 'uses'"
                    )


def _analyze_workflow_file(
    workflow_file: str,
    max_retries: int,
    retry_delay: float,
    report: bool = False,
) -> FileAnalysisResult:
    last_error = "Unknown error"

    for attempt in range(1, max_retries + 1):
        try:
            workflow = Workflow.from_file(workflow_file)
            _validate_workflow_contract(workflow)
            runner = Runner(workflow)
            smells = runner.run_all()
            smell_rows = util.smells_to_csv_rows(
                workflow.name, workflow.file_content, smells
            )
            styling_rows = util.styling_to_csv_rows(
                workflow.name, workflow.file_content, workflow.styling
            )

            category_counts = Counter()
            for row in smell_rows:
                category_counts[row[1]] += 1

            report_lines = None
            if report:
                report_lines = _build_workflow_report_lines(workflow, smells)

            return FileAnalysisResult(
                file_path=workflow_file,
                rows=smell_rows + styling_rows,
                smell_count=len(smells),
                category_counts=category_counts,
                attempts=attempt,
                report_lines=report_lines,
            )
        except NonRetriableWorkflowError as exc:
            return FileAnalysisResult(
                file_path=workflow_file,
                rows=[],
                smell_count=0,
                category_counts=Counter(),
                attempts=attempt,
                error=str(exc),
                non_retriable=True,
            )
        except ValueError as exc:
            return FileAnalysisResult(
                file_path=workflow_file,
                rows=[],
                smell_count=0,
                category_counts=Counter(),
                attempts=attempt,
                error=str(exc),
                non_retriable=True,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries and retry_delay > 0:
                time.sleep(retry_delay)

    return FileAnalysisResult(
        file_path=workflow_file,
        rows=[],
        smell_count=0,
        category_counts=Counter(),
        attempts=max_retries,
        error=last_error,
    )


def _print_batch_summary(
    total_files: int,
    successful_files: int,
    failed_files: list[tuple[str, str]],
    total_smells: int,
    total_retries: int,
    files_with_findings: int,
    non_retriable_failures: int,
    category_counts: Counter,
    file_issue_counts: Counter,
    elapsed_seconds: float,
) -> None:
    typer.echo("\nBatch summary")
    typer.echo(f"- Files found: {total_files}")
    typer.echo(f"- Files processed successfully: {successful_files}")
    typer.echo(f"- Files failed: {len(failed_files)}")
    typer.echo(f"- Non-retriable failures: {non_retriable_failures}")
    typer.echo(f"- Total retries performed: {total_retries}")
    typer.echo(f"- Files with findings: {files_with_findings}")
    typer.echo(f"- Total smells detected: {total_smells}")
    typer.echo(f"- Total execution time: {elapsed_seconds:.2f}s")
    if total_files > 0:
        typer.echo(f"- Success rate: {(successful_files / total_files) * 100:.2f}%")
    if successful_files > 0:
        typer.echo(
            f"- Avg time per successful file: {elapsed_seconds / successful_files:.2f}s"
        )

    if len(category_counts) > 0:
        typer.echo("- Smells by category:")
        for category, count in category_counts.most_common():
            typer.echo(f"  - {category}: {count}")

    if len(file_issue_counts) > 0:
        typer.echo("- Top 5 files with most findings:")
        for file_name, count in file_issue_counts.most_common(5):
            typer.echo(f"  - {file_name}: {count}")

    if len(failed_files) > 0:
        typer.echo("- Failed files:")
        for file_path, error_message in failed_files:
            typer.echo(f"  - {file_path}: {error_message}")


@app.command(name="all")
def analyze_all(
    workflow_folder: Annotated[str | None, typer.Argument()] = None,
    csv_output: Annotated[str | None, typer.Option("--csv-output", "-o")] = None,
    max_retries: Annotated[
        int, typer.Option("--max-retries", min=1, help="Maximum retries per file.")
    ] = 3,
    retry_delay: Annotated[
        float,
        typer.Option("--retry-delay", min=0.0, help="Delay in seconds before retry."),
    ] = 1.0,
    report_each_file: Annotated[
        bool,
        typer.Option(
            "--report-each-file/--no-report-each-file", help="Show per-file findings."
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Maximum number of files processed concurrently.",
        ),
    ] = 8,
    summary: Annotated[
        bool, typer.Option("--summary/--no-summary", help="Print final batch summary.")
    ] = True,
) -> None:

    if workflow_folder is None:
        workflow_folder = "./.github/workflows"

    workflow_files = [
        join(workflow_folder, f)
        for f in listdir(workflow_folder)
        if isfile(join(workflow_folder, f))
        and (os.path.splitext(f)[1] == ".yml" or os.path.splitext(f)[1] == ".yaml")
    ]

    if csv_output is None:
        csv_output = "gha_smells.csv"
    output_path = _resolve_output_path(csv_output)
    util.write_csv_header(output_path)

    start_time = time.time()
    pending_rows: list[list[str]] = []
    total_rows_written = 0
    failed_files: list[tuple[str, str]] = []
    successful_files = 0
    total_smells = 0
    total_retries = 0
    files_with_findings = 0
    non_retriable_failures = 0
    category_counts = Counter()
    file_issue_counts = Counter()
    processed_since_flush = 0
    flush_every = 100

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _analyze_workflow_file,
                wf,
                max_retries,
                retry_delay,
                report_each_file,
            ): wf
            for wf in workflow_files
        }

        with tqdm(
            total=len(workflow_files), desc="Processing workflows", unit="file"
        ) as progress:
            for future in as_completed(futures):
                result = future.result()
                progress.update(1)

                if result.error is not None:
                    failed_files.append((result.file_path, result.error))
                    if result.non_retriable:
                        non_retriable_failures += 1
                        tqdm.write(
                            f"Failed to process {result.file_path}: non-retriable input error: {result.error}"
                        )
                    else:
                        tqdm.write(
                            f"Failed to process {result.file_path} after {result.attempts} attempts: {result.error}"
                        )
                else:
                    pending_rows.extend(result.rows)
                    successful_files += 1
                    total_smells += result.smell_count
                    total_retries += max(0, result.attempts - 1)
                    if result.smell_count > 0:
                        files_with_findings += 1
                    category_counts.update(result.category_counts)
                    file_issue_counts[os.path.basename(result.file_path)] += len(
                        result.rows
                    )

                    if report_each_file and result.report_lines is not None:
                        for line in result.report_lines:
                            tqdm.write(line)

                processed_since_flush += 1
                if processed_since_flush >= flush_every:
                    if pending_rows:
                        util.append_smells_csv(output_path, pending_rows)
                        total_rows_written += len(pending_rows)
                        pending_rows = []
                    processed_since_flush = 0

                progress.set_postfix(
                    {
                        "ok": successful_files,
                        "err": len(failed_files),
                        "invalid": non_retriable_failures,
                        "retry": total_retries,
                        "smells": total_smells,
                        "csv": total_rows_written + len(pending_rows),
                    }
                )

    if pending_rows:
        util.append_smells_csv(output_path, pending_rows)
        total_rows_written += len(pending_rows)
        pending_rows = []

    typer.echo(f"CSV report saved to {output_path} ({total_rows_written} rows)")

    elapsed_seconds = time.time() - start_time
    if summary:
        _print_batch_summary(
            total_files=len(workflow_files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_smells=total_smells,
            total_retries=total_retries,
            files_with_findings=files_with_findings,
            non_retriable_failures=non_retriable_failures,
            category_counts=category_counts,
            file_issue_counts=file_issue_counts,
            elapsed_seconds=elapsed_seconds,
        )


@app.command(name="file")
def analyze_one(
    file_path: str = typer.Argument(),
    csv_output: Annotated[str | None, typer.Option("--csv-output", "-o")] = None,
) -> None:
    workflow = Workflow.from_file(file_path)
    smells = analyze_and_report_workflow(workflow)
    rows = util.smells_to_csv_rows(workflow.name, workflow.file_content, smells)
    rows.extend(
        util.styling_to_csv_rows(workflow.name, workflow.file_content, workflow.styling)
    )

    if csv_output is None:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        csv_output = f"{filename}_smells.csv"

    output_path = _resolve_output_path(csv_output)
    util.write_smells_csv(output_path, rows)
    typer.echo(f"CSV report saved to {output_path}")
