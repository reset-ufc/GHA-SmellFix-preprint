# GHA

`gha-ci-detector`, the smell detector of the catalog of Khatami et al. The study keeps it fixed as
its baseline: Step 09 of the dataset reads its scan of the corpus (`data/gha_smells.csv`), and the
robustness analysis runs it on the semi-synthetic workflows. The study's definitions of the 19
smells are in [`docs/smell-catalog.md`](../../docs/smell-catalog.md).

## Setup

`uv sync` at the repository root installs the `gha-ci-detector` command.

## Run

```bash
uv run gha-ci-detector all data/downloaded_workflows/   # every workflow of a folder -> CSV
uv run gha-ci-detector file path/to/workflow.yml        # one workflow
```

`all` without a folder reads `./.github/workflows`. It processes the files in parallel, retries
failed files, and prints a summary.

| Option of `all` | Default | Meaning |
|-----------------|---------|---------|
| `--csv-output`, `-o` | `gha_smells.csv` | the output CSV |
| `--workers`, `-w` | `8` | files processed at the same time |
| `--max-retries` | `3` | attempts per failed file |
| `--retry-delay` | `1.0` | seconds before each retry |
| `--report-each-file` | off | prints the findings of each file |
| `--summary` | on | prints the final summary |

`file` takes `--csv-output` (default `<filename>_smells.csv`). `--version` prints the version.

## Checks

```bash
uv run ruff check --no-fix modules/gha_detector
```
