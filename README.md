# GHA-SmellFix

Replication package of the paper on GHA-SmellFix, an LLM pipeline that detects and refactors
configuration smells in GitHub Actions workflows. The pipeline has two orchestration designs:
**D1**, joint (one agent detects the target smell and, when it is present, returns the patch in the
same answer), and **D2**, decomposed (a detection agent decides first, and a refactoring agent runs
only when the smell is detected). The package builds the ground truth, runs both designs with four
LLMs, and writes the tables of the paper from the stored runs.

> **Data.** The data archive, `data.tar.xz` (502 MB, 44,964 files, SHA-256
> `d806309f1f2536cd46f3f8a921fbd334182c6e7bf10c3507b01e29a1496bd3c1`), is in this repository,
> stored with [Git LFS](https://git-lfs.com/).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). The root `pyproject.toml` is a uv workspace with
the four `modules/*` as members, so one command installs every module from `uv.lock`:

```bash
uv sync
```

A clone made with Git LFS installed already has `data.tar.xz`; otherwise install Git LFS and run
`git lfs pull`. Extract the archive at the repository root and check every file against its
manifest:

```bash
git lfs pull
tar -xJf data.tar.xz
sha256sum -c data/SHA256SUMS
```

The notebooks run in Jupyter or VS Code with the `replication` kernel:

```bash
uv run python -m ipykernel install --user --name study --display-name "replication"
```

## Run

Run every command from the repository root. Each stage reads what the previous one wrote.

| Stage | Module | Command | Writes |
|-------|--------|---------|--------|
| 1. Ground truth | [`modules/dataset/`](modules/dataset/README.md) | notebooks `01` to `13` | `data/13_final_ground_truth.json` and its variants |
| 2. Agent runs | [`modules/agents_pipeline/`](modules/agents_pipeline/README.md) | `uv run python -m gha_agents.experiment <config>` | one result JSON per run |
| 3. Analysis | [`modules/analysis/`](modules/analysis/README.md) | notebooks `00` to `06` | `data/artifacts/` |

The ground truth and the runs of the paper ship in `data/`. Stage 3 alone rewrites every table and
needs no API key; stages 1 and 2 are only needed to rebuild the ground truth or to call the models
again. [`modules/gha_detector/`](modules/gha_detector/README.md) is GHA (`gha-ci-detector`), the
detector of the smell catalog of Khatami et al. Stage 1 reads its scan of the corpus, and the
robustness analysis of stage 3 runs it on the semi-synthetic workflows.

To run D1 or D2 on a workflow of your own, with no ground truth, use `gha-smellfix`; its inputs and
flags are in the [agents pipeline README](modules/agents_pipeline/README.md#run-on-one-workflow):

```bash
uv run gha-smellfix path/to/ci.yml --smell MissingJobTimeout --design D1 --model haiku45
```

The workflow-level feasibility diagnostic follows its own protocol,
[`modules/agents_pipeline/docs/workflow_level_feasibility_detection_refactoring_protocol.md`](modules/agents_pipeline/docs/workflow_level_feasibility_detection_refactoring_protocol.md),
and its steps are in the [agents pipeline README](modules/agents_pipeline/README.md#workflow-level-feasibility-diagnostic).

## Outputs

| Paper | File | Notebook |
|-------|------|----------|
| `tab:evaluated_llms` | `data/artifacts/evaluated_llms.tex` | `03_tables` |
| `tab:rq1_overall_detection` | `data/artifacts/rq1_overall_detection.tex` | `01_detection_effectiveness` |
| `tab:rq1_smell_level_detection_effectiveness` | `data/artifacts/rq1_smell_level_detection_effectiveness.tex` | `01_detection_effectiveness` |
| `tab:rq2_refactoring_outcomes_by_smell` | `data/artifacts/rq2_refactoring_outcomes_by_smell.tex` | `02_refactoring_effectiveness` |
| `tab:rq2_end_to_end_conditional` | `data/artifacts/rq2_end_to_end_conditional.tex` | `02_refactoring_effectiveness` |
| `tab:rq2_clean_removal_by_model` | `data/artifacts/rq2_clean_removal_by_model.tex` | `02_refactoring_effectiveness` |
| `tab:rq2_negative_edit_types` | `data/artifacts/rq2_negative_edit_types.tex` | `02_refactoring_effectiveness` |
| `tab:rq3_absolute_overhead` | `data/artifacts/rq3_absolute_overhead.tex` | `03_tables` |
| `tab:rq3_paired_tests_across_configurations` | `data/artifacts/rq3_paired_tests_across_designs.tex` | `03_tables` |

| Paper paragraph (*Robustness and Diagnostic Analyses*) | File |
|---------------------------------------------------------|------|
| *GHA-based candidate retrieval* | `data/artifacts/robustness/gha_retrieval.json` |
| *Balanced evaluation* | `data/artifacts/robustness/prevalence_adjusted_precision.json` |
| *Semi-synthetic UnsafeDeployOnFork* | `data/artifacts/robustness/semi_synthetic.json` |
| *Workflow-level feasibility diagnostic* | `data/artifacts/workflow_feasibility/` |

The 19 smells, as the study defines them, are in [`docs/smell-catalog.md`](docs/smell-catalog.md).
The annotators' guides are [`docs/validation/detection-guide.md`](docs/validation/detection-guide.md)
and [`docs/validation/refactoring-codebook.md`](docs/validation/refactoring-codebook.md). Their
notes are in [`docs/validation/notes-detection.csv`](docs/validation/notes-detection.csv) for the
detection labels of `data/11_manual_evaluation.csv` and in the `notes` column of
`data/validation/verdicts_raters.csv` for the refactoring verdicts. The prompts of each design are
in [`modules/agents_pipeline/docs/prompts/`](modules/agents_pipeline/docs/prompts/).

## Layout

```
pyproject.toml, uv.lock    uv workspace and locked environment
data.tar.xz                data archive, stored with Git LFS
data/                      extracted from data.tar.xz: ground truth, corpus, runs, artifacts
docs/                      smell catalog and annotation guides
modules/
  dataset/                 stage 1: notebooks 01 to 13 build the ground truth
  agents_pipeline/         stage 2: D1 and D2, one result JSON per run
  analysis/                stage 3: metrics, tables and robustness analyses
  gha_detector/            GHA, the smell detector of the catalog
```

## Checks

```bash
uv run pytest
```
