# Analysis

Stage 3 of the package: the metrics of the paper. The code reads the agent runs and the reference
inputs under `data/`, and writes the paper's tables, the robustness analyses, and the CSV files
behind them to `data/artifacts/`. It makes no API call. The notebooks only orchestrate; the logic is
in `utils/`.

## Setup

From the repository root, `uv sync` installs this module with the rest of the workspace. The data
comes from `data.tar.xz` (see the [root README](../../README.md#setup)). Without uv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r modules/analysis/requirements.txt
pip install -e modules/gha_detector
```

## Run

Run the notebooks in order, from the repository root, in Jupyter or VS Code with the `replication`
kernel, or headless:

```bash
uv run pytest --nbmake --nbmake-timeout=3600 modules/analysis/notebooks/0*.ipynb
```

`02_construct_build` must run before `02_refactoring_effectiveness`, which reads what it writes.

| Notebook | Writes |
|----------|--------|
| `00_dataset` | nothing: loads the runs and checks their columns |
| `01_detection_effectiveness` | `rq1_overall_detection.tex`, `rq1_smell_level_detection_effectiveness.tex`, `rq1_accuracy_spread.csv` (accuracy of each repetition and its spread) |
| `02_construct_build` | `constructs/` (construct diffs per patch), `validation/` (behavior index and validation scope) |
| `02_refactoring_effectiveness` | `rq2_refactoring_outcomes_by_smell.tex`, `rq2_end_to_end_conditional.tex`, `rq2_clean_removal_by_model.tex`, `rq2_negative_edit_types.tex` |
| `03_tables` | `evaluated_llms.tex`, `rq3_absolute_overhead.tex`, `rq3_paired_tests_across_designs.tex` |
| `04_cost_latency` | `cost_latency/` (CSVs and figures), `cost_latency.tex`, `stats_latency.tex`, `stats_tokens.tex` |
| `05_validation` | nothing: prints Cohen's κ and the disagreements of the manual validation |
| `06_robustness` | `robustness/`: GHA-based candidate retrieval, prevalence-adjusted precision, semi-synthetic UnsafeDeployOnFork |

Every path is under `data/artifacts/`. `STUDY_DATA_ROOT` reads the runs from another directory
(default `data/agents_runs_outputs/`); the reference inputs stay under `data/`.

The workflow-level feasibility diagnostic has its own command, described in the
[agents pipeline README](../agents_pipeline/README.md#workflow-level-feasibility-diagnostic):

```bash
uv run python -m utils.workflow_feasibility
```

## Inputs

| File under `data/` | Content |
|--------------------|---------|
| `agents_runs_outputs/output-final-{deepseek,gemini-n15,gpt-n15,haiku-n15}/` | the runs of the paper, one result JSON per run, under `design_d1/` and `design_d2/` |
| `ground_truth_n15.json` | the 570 pairs every model is scored on (285 present, 285 absent), from `modules/agents_pipeline/scripts/split_unsafedeploy_for_agents.py` |
| `validation/` | the manual verdicts on the patches: each rater, the disagreements, the resolved verdicts |
| `constructs_catalogue.csv`, `smell_expected_construct_mapping_refined.csv` | the workflow constructs and the ones each smell's fix is expected to touch |
| `github-workflow.json` | the GitHub Actions JSON schema |
| `11_manual_evaluation.csv`, `manual_evaluation_gha_overturned.csv` | the manual labels of every candidate: the ones that agree with GHA, and the ones that contradict it |
| `9_ground_truth_candidates.csv`, `9b_loc_cap.json`, `step10_report.json` | the candidate pool the ground truth was sampled from |
| `12d_synth_UnsafeDeployOnFork_present.json`, `12e_sinth_manual_evaluation.csv`, `13_final_ground_truth.json` | the semi-synthetic UnsafeDeployOnFork workflows, their labels, and their base workflows |

## Decisions the paper does not state

- **Runs.** Every table reads the first repetition (`repetition == 1`); only the accuracy spread
  reads the three repetitions of GPT-5.4-nano and Claude Haiku 4.5. Runs outside the 570 pairs
  are left out: DeepSeek V4 Flash ran all 1,710 pairs of the ground truth. The loader drops Gemma 4,
  which is not scored on the same pairs, and keeps one run per model, design, repetition,
  workflow and smell.
- **Truth.** Detection is scored against `data/ground_truth_n15.json`, not against the
  `ground_truth` field that each run records.
- **Metrics.** scikit-learn, with *present* as the positive class; a zero denominator gives no
  value. The Avg. row of `tab:rq1_overall_detection` is the mean of the four models, and the Avg.
  row of `tab:rq1_smell_level_detection_effectiveness` is the mean over the smells. Its gray rows are
  the five smells with the lowest mean F1.
- **Refactoring outcomes.** Each run counts on its own, with no majority over the eight runs of a
  pair. A positive run is Clean Removal when the smell is removed, the manual
  `potential_behavior_impact` answer is `no`, the constructs are the expected ones, and the patch
  passes the schema; Removal with Potential Behavior-Impacting Change when the smell is removed and
  that answer is `yes`, whatever the other checks give; Removal with Unexpected Changes when the
  smell is removed and that answer is `no`, but the constructs or the schema check fail; Failed or No Patch otherwise, detection misses
  included. A run with an `uncertain` value in any manual field is left out of every count. The
  gray rows of `tab:rq2_refactoring_outcomes_by_smell` are the five smells with the lowest Clean
  Removal, ties broken toward more Removal with Potential Behavior-Impacting Change, then more
  Failed or No Patch.
- **Schema check.** `patch_valid` validates the whole patched workflow against the GitHub Actions
  schema, so an error that the original workflow already had also fails it.
- **Unnecessary edits.** A negative run is an Unnecessary Edit when its manual verdict is
  `unnecessary`. Every other assessable negative run is a Correct Abstention, including a
  `necessary` verdict and a run whose model found the smell absent, even if that D1 run returned a
  changed workflow. The type of an Unnecessary Edit is the riskiest construct it changes: a
  `timeout-minutes` change is localized, a naming change is cosmetic, and every other change is
  behavior-relevant.
- **Paired tests.** The unit is the (workflow, smell) pair, with one D1 and one D2 run of the same
  model. Binary outcomes use McNemar's exact test; latency and cost use the paired Wilcoxon
  signed-rank test, without the zero differences. Holm corrects over the four models of one
  outcome, and a test is significant when its corrected p is below 0.05. The effect is D2 minus
  D1: the difference of the rates, or the median of the paired differences.
- **Cost.** The estimated cost prices the input tokens and the output tokens at each model's list
  price (`utils/cost_latency.py`, `PRICING`). The reasoning tokens count as output where the
  provider reports them apart.
- **GHA-based candidate retrieval.** A GHA-flagged candidate is confirmed when its manual label is
  `present`, whatever its number of instances.
- **Prevalence-adjusted precision.** The pool is the workflows that Step 10 sampled from. A smell's
  prevalence combines the share of workflows that GHA flags with the confirmation rate of the
  flagged candidates and the miss rate of the unflagged ones; the pool prevalence is the mean over
  the 19 smells. It adjusts the precision of each model and design from its pooled recall and
  false-positive rate, and a design's value is the mean of the four models.
- **Semi-synthetic UnsafeDeployOnFork.** The exclusion drops the smell's 30 pairs of the 570 and
  repeats each analysis with the loaders and rules of the table it comes from.

## Layout

```
notebooks/                       00 to 06, orchestration only
utils/
  dataloader.py                  result JSONs -> one DataFrame
  designs.py                     the designs D1 and D2
  detection.py                   RQ1 metrics and the n15 labels
  construct_diff.py, constructs.py, schema_validity.py, refactored_instances.py, validation_scope.py
                                 RQ2: constructs of each patch, schema check, validation scope
  stats.py                       McNemar, Wilcoxon, Holm, Cliff's delta
  cost_latency.py                tokens, cost and latency
  inter_rater.py, validation_summary.py
                                 report of the manual validation
  gha_retrieval.py               GHA's flags against the manual labels
  prevalence_adjusted_precision.py
                                 precision under the candidate-pool prevalence
  semi_synthetic.py              GHA check and exclusion of the semi-synthetic smell
  workflow_feasibility.py        the workflow-level feasibility diagnostic
  tables/                        one LaTeX generator per table of the paper
tests/
```

The construct catalogue in `data/constructs_catalogue.csv` is from Talebzadeh et al., *On the
GitHub Actions Language: Usage, Evolution, and Workflow Reliability*; cite that work when reusing
it.

## Checks

```bash
uv run pytest modules/analysis
```
