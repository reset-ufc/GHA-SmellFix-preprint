# Dataset

Stage 1 of the package: the study's dataset and its human-validated ground truth of
(workflow, smell) pairs. Each step is a numbered notebook that writes under `data/`; Step 08 is
also a script. Every notebook opens with a cell that lists its inputs and outputs.

## Setup

`uv sync` at the repository root installs this module, and the `replication` kernel runs the
notebooks (see the [root README](../../README.md#setup)). The first cell of each notebook moves to
the repository root, so a notebook runs from any directory.

One input ships compressed. Decompress it once, from the repository root:

```bash
unxz -k data/gha_smells.csv.xz   # Step 09, about 2.2 GB
```

The YAML corpus, `data/downloaded_workflows/`, is extracted from `downloaded_workflows.tar.xz` on
first use, after a SHA-256 check (`dataset_utils.bootstrap.ensure_extracted`).

## Run

Run the steps in order, from the repository root: each step checks its input against the report of
the previous one and stops on a mismatch.

```bash
uv run pytest --nbmake --nbmake-timeout=3600 \
  modules/dataset/01_clean_repositories.ipynb \
  modules/dataset/03_merge_datasets.ipynb \
  modules/dataset/04_final_cleaning.ipynb \
  modules/dataset/05_refine_columns.ipynb \
  modules/dataset/06_classify_bot_authors.ipynb \
  modules/dataset/07_offline_apply_criteria.ipynb \
  modules/dataset/08b_final_report.ipynb \
  modules/dataset/09_build_ground_truth_candidates.ipynb \
  modules/dataset/09b_compute_loc_cap.ipynb \
  modules/dataset/10_sample_ground_truth.ipynb \
  modules/dataset/10b_ground_truth_resample.ipynb \
  modules/dataset/11_summarize_evaluation.ipynb \
  modules/dataset/12_human_validated_ground_truth.ipynb \
  modules/dataset/12b_balanced_ground_truth.ipynb \
  modules/dataset/12c_synthesis_targets.ipynb \
  modules/dataset/12e_recalc_kappa.ipynb \
  modules/dataset/13_final_ground_truth.ipynb
```

| Step | Notebook | Does |
|------|----------|------|
| 01 | `01_clean_repositories` | cleans the repository metadata |
| 02 | `02_deduplicate_workflows` | deduplicates the workflow history; optional, no step reads it |
| 03 | `03_merge_datasets` | joins each workflow to its repository |
| 04 | `04_final_cleaning` | types, UTC dates, rows with missing or inverted fields |
| 05 | `05_refine_columns` | the columns kept for the analysis |
| 06 | `06_classify_bot_authors` | flags bot authors |
| 07 | `07_offline_apply_criteria` | the inclusion and exclusion criteria that need no network; the cutoff dates are fixed |
| 08 | `08_online_enrich_dataset`, `scripts/08_online_enrich_dataset.py` | GitHub API data and the workflow corpus; **frozen** |
| 08b | `08b_final_report` | summary statistics of the enriched dataset |
| 09 | `09_build_ground_truth_candidates` | crosses the GHA scan with the corpus: the candidate pool |
| 09b | `09b_compute_loc_cap` | the size cap of a workflow (Tukey upper fence on its lines) |
| 10 | `10_sample_ground_truth` | the annotation pool per smell and polarity |
| 10b | `10b_ground_truth_resample` | more annotation targets, without the ones of Step 10 |
| 11 | `11_summarize_evaluation` | reads the manual labels; Cohen's κ |
| 12 | `12_human_validated_ground_truth` | the unbalanced ground truth; stops on a contradictory label |
| 12b | `12b_balanced_ground_truth` | as many positives as negatives per smell: `12b_final_ground_truth.json` and the n15 variants |
| 12c | `12c_synthesis_targets` | the 50 negatives of UnsafeDeployOnFork that the synthesis mutates |
| 12d | `12d_synthesize_positives` | checks the hash of the semi-synthetic positives; **frozen** |
| 12e | `12e_recalc_kappa` | Cohen's κ with the labels of the semi-synthetic positives |
| 13 | `13_final_ground_truth` | the final ground truth, `13_final_ground_truth.json` |

`_11b_latex_tables.ipynb` writes the agreement tables of Step 11 in LaTeX, and is optional.

## Frozen steps

Steps 08 and 12d do not give the same output twice, so they are never re-run. Step 08 reads the
live GitHub API, needs a token, and records timestamps; Step 12d asked an LLM for the
semi-synthetic positives, with no fixed seed. Their outputs ship in `data/`, and the later steps
check them by hash. Every other step is deterministic.

## Inputs from outside this module

- **Workflow histories** (Steps 01 to 03): the dataset of Cardoen et al. (2024).
- **GHA scan** (Step 09): `data/gha_smells.csv`, the output of GHA over the corpus.
  `data/gha_analyzer_exclusions.txt` lists the workflows that GHA cannot parse: Step 09 leaves them
  out of the candidate pool, and stops if the scan misses any other workflow.
- **Manual labels** (Step 11): `data/11_manual_evaluation.csv`, exported from the annotation tool,
  which this package does not include. It keeps the candidates whose label agrees with GHA's flag
  (`tool_output`); the others are in `data/manual_evaluation_gha_overturned.csv`, which only the
  robustness analysis reads. The annotators' guides are in [`docs/validation/`](../../docs/validation/).
- **Semi-synthetic positives** (Steps 12d and 12e): generated by a separate synthesis tool, then
  labeled by hand into `data/12e_sinth_manual_evaluation.csv`.

## Verify

After a run, the rebuilt ground truth must match the shipped hashes, which are the ones of the
ground truth that the runs used:

```bash
grep -E '  data/(10_ground_truth_sample|12_ground_truth_unbalanced|12b_final_ground_truth|12b_ground_truth_n15|12c_synthesis_targets_UnsafeDeployOnFork_absent|13_final_ground_truth)\.json$' \
  data/SHA256SUMS | sha256sum -c
```

Steps 01 and 03 do not pin their row order, so the hashes in their reports may change with the
library versions. No later step depends on that order.

## Layout

```
NN_*.ipynb          the steps
dataset_utils/      paths, hashing, integrity checks, notebook bootstrap, sampling, balancing,
                    synthetic merge, inter-rater agreement, LaTeX
scripts/            Step 08 as a script
tests/              tests of the helpers
```

## Checks

```bash
uv run pytest modules/dataset/tests
```
