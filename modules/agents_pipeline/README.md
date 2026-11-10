# Agents pipeline

Stage 2 of the package: the two designs of GHA-SmellFix. For each (workflow, target smell) pair of
a ground truth, a run asks one model to detect the smell and, when it is present, to patch the
workflow, and writes one result JSON. It computes no metric; `modules/analysis/` does.

- **D1** (`D1Workflow`): one agent detects the smell and refactors in the same answer.
- **D2** (`D2Workflow`): a detection agent decides first; a refactoring agent runs only when the
  smell is detected. When it is not, the run returns the original workflow unchanged.

The models are called through [Agno](https://github.com/agno-agi/agno), reached only through
`AgentInvoker`; each design drives its agents in plain Python (`gha_agents/teams.py`). The prompts
of each design are in [`docs/prompts/`](docs/prompts/), built by `gha_agents/prompts.py`. Before
the paper named the designs D1 and D2, they were called C1 and C2; some data may still use the old
names.

## Setup

`uv sync` at the repository root installs this module. The API keys go in
`modules/agents_pipeline/.env`, which the pipeline reads wherever it runs from:

```bash
cp modules/agents_pipeline/.env.example modules/agents_pipeline/.env
```

| Model (alias) | Provider | Key |
|---------------|----------|-----|
| Claude Haiku 4.5 (`haiku45`) | `anthropic` | `ANTHROPIC_API_KEY` |
| GPT-5.4-nano (`gpt54-nano`) | `openai` | `OPENAI_API_KEY` |
| Gemini 3.5 Flash (`gemini35-flash`) | `vertex` | no key: `gcloud auth application-default login`, then `GOOGLE_CLOUD_PROJECT` (and `GOOGLE_CLOUD_LOCATION`, default `us-central1`) |
| DeepSeek V4 Flash (`deepseek-v4-flash`) | `openrouter` | `OPENROUTER_API_KEY` |

A provider that no config routes to can stay blank.

## Run

Run from the repository root: the configs read `./data/...`. A run calls the models and costs money;
`--dry-run` prints the expanded runs and calls nothing:

```bash
uv run python -m gha_agents.experiment modules/agents_pipeline/configs/n15.yaml --dry-run
uv run python -m gha_agents.experiment modules/agents_pipeline/configs/n15.yaml
```

| Flag | Default | Meaning |
|------|---------|---------|
| `CONFIG` | required | the experiment YAML |
| `--dry-run` | off | prints the runs, sends nothing |
| `--max-workers N` | from the YAML | runs in parallel |
| `--timeout S` | from the YAML | HTTP timeout of each model call |
| `--progress` | off | live progress bars |
| `--skip-preflight` | off | skips the gate that probes each model before the runs |
| `--reconcile` | off | when the ground truth changed, re-runs only the affected cells instead of stopping |
| `--verbose`, `-V` | off | HTTP logs on the console; the run log always has them |

| Config | Ground truth | Models and repetitions | Writes to |
|--------|--------------|------------------------|-----------|
| `n15.yaml` | `data/12b_ground_truth_n15.json` | GPT-5.4-nano and Claude Haiku 4.5 x 3, Gemini 3.5 Flash x 1 | `./output-n15` |
| `full.yaml` | `data/12b_final_ground_truth.json` | Claude Haiku 4.5 x 3, Gemini 3.5 Flash and DeepSeek V4 Flash x 1 | `./output-full` |
| `synth_missingperm.yaml` | `data/12b_ground_truth_n15_MissingPermExternalActions.json` | as `n15.yaml` | `./output-synth-missingperm` |
| `synth_unsafedeploy_n15.yaml` | `data/12f_ground_truth_n15fill_UnsafeDeployOnFork.json` | as `n15.yaml` | `./output-synth-unsafedeploy-n15` |
| `synth_unsafedeploy_full.yaml` | `data/12f_unsafedeploy_synth90_full.json` | DeepSeek V4 Flash x 1 | `./output-synth-unsafedeploy-full` |
| `workflow_feasibility.yaml` | `data/workflow_feasibility/ground_truth.json` | Claude Haiku 4.5 x 1 | `data/workflow_feasibility/runs/output-haiku45` |

Each config runs D1 and D2. The runs of the paper are in
`data/agents_runs_outputs/output-final-{deepseek,gemini-n15,gpt-n15,haiku-n15}/`: one folder per
model, merged from these outputs with `scripts/merge_outputs.py`, whose docstring has the command.
These folders also hold 154 runs of GPT-5.4-nano, Claude Haiku 4.5 and Gemini 3.5 Flash on 11 pairs
outside the 570 that the analysis scores (`data/ground_truth_n15.json`), made to match the pairs of
an earlier run.

`scripts/split_unsafedeploy_for_agents.py` writes the ground truths of the two `synth_unsafedeploy`
configs from the dataset's final ground truth, and `data/ground_truth_n15.json`: the 570 pairs of
`n15.yaml` and `synth_unsafedeploy_n15.yaml` together, which the analysis scores.

```bash
uv run python modules/agents_pipeline/scripts/split_unsafedeploy_for_agents.py
```

## Output

```
<output_dir>/design_d{1,2}/
  results/<slug>.json            one result per run
  patched_workflows/<slug>.yml   the workflow the run returns
  traces/<slug>.json             prompts, answers, reasoning and tokens of every model call
  error.log                      one JSON line per failed run
  error_artifacts/               raw answers that broke the output contract
  <timestamp>.log                the execution log
```

`<slug>` is `<workflow>@<D1|D2>@rep<k>@<smell>@<model>`, plus `@synth` for a semi-synthetic
workflow. The main fields of a result:

| Field | Content |
|-------|---------|
| `file_key`, `target_smell`, `ground_truth` | the pair and its label in the ground truth the run read |
| `design`, `model_alias`, `model_ids`, `repetition` | what ran |
| `smell_detected`, `count`, `evidence_ranges`, `detection_justification` | the detection |
| `patched_workflow_path`, `refactor_justification` | the patch |
| `patch_valid`, `patch_validation_error` | the GitHub Actions schema check of the patched workflow |
| `sampling_settings`, `prompt_hashes` | the sampling and the hash of each prompt |
| `token_usage`, `agent_call_count`, `run_duration_seconds` | the effort |

## Run on one workflow

`gha-smellfix` runs one design on one workflow for one target smell, and needs no ground truth. Its
inputs are those of the paper: the workflow, the target smell and the repository context. It runs
the agents, prompts and sampling of the runs above, with the keys of *Setup*, and costs money:

```bash
uv run gha-smellfix path/to/ci.yml --smell MissingJobTimeout --design D2 --model haiku45 \
  --repository owner/repo --default-branch main --file-path .github/workflows/ci.yml \
  --out ci.patched.yml
```

| Flag | Default | Meaning |
|------|---------|---------|
| `WORKFLOW` | required | the workflow YAML |
| `--smell` | required | the target smell, as named in [`docs/smell-catalog.md`](../../docs/smell-catalog.md) |
| `--design` | required | `D1` or `D2` |
| `--model` | required | an alias of the table in *Setup* |
| `--repository` | none | repository context: `owner/repo` |
| `--default-branch` | none | repository context: the default branch |
| `--file-path` | none | repository context: the workflow's path in the repository |
| `--out FILE` | none | writes the workflow the run returns |
| `--json FILE` | none | writes the result, with the fields of the table above; `patched_workflow` holds the workflow itself |
| `--timeout S` | 180 | HTTP timeout of each model call |

It prints the verdict, the justifications, the token usage and the diff of the patch. A repository
flag left out is left out of the prompt. The exit code is 1 when the run fails, and 2 when an input
is wrong.

## Decisions the paper does not state

- **Sampling.** Temperature 0 where the model accepts it. Claude Haiku 4.5 runs with extended
  thinking (2,048-token budget), and GPT-5.4-nano with `reasoning_effort: medium`; their APIs
  reject temperature 0 then, so both keep the provider's default, and Claude's `top_p` is 0.95.
- **Preflight.** Before the runs, each model gets one probe: the key works, the model returns
  reasoning, and the temperature is the one above. A failure stops the command.
- **Resume.** A run whose result JSON exists, and its trace when traces are on, is skipped, so the
  same command continues an interrupted execution.
- **Failures.** A failed run writes no result: it goes to `error.log`, and resume tries it again.
- **Count.** `count` is diagnostic. Presence is `smell_detected`.
- **Cost.** No result stores a cost; the analysis derives it from `token_usage` and `model_ids`.

## Workflow-level feasibility diagnostic

The diagnostic follows
[`docs/workflow_level_feasibility_detection_refactoring_protocol.md`](docs/workflow_level_feasibility_detection_refactoring_protocol.md):
8 real workflows, each annotated for all 19 smells, one run per (workflow, smell) and design with
Claude Haiku 4.5, 304 runs. Its files are under `data/workflow_feasibility/`, in `data.tar.xz`.
Run every step from the repository root.

1. **Select** the workflows. Done: `select` wrote `selected_workflows.json` and the annotation
   sheet, one tab per annotator.
2. **Merge** the two exports (`annotator_a.csv`, `annotator_b.csv`). Agreements are kept; a
   disagreement leaves `manual_label` blank, with both verdicts beside it. Then resolve each blank
   row by hand.

   ```bash
   uv run python modules/agents_pipeline/scripts/workflow_feasibility.py consensus
   ```

3. **Freeze** the hand-made files: `manifest.json` records the SHA-256, size and completion time
   (UTC) of each, and a test fails while a file no longer matches its entry. Run it again after the
   patch review of step 6.

   ```bash
   uv run python modules/agents_pipeline/scripts/workflow_feasibility.py freeze
   ```

4. **Build** the ground truth. It prints the agreement before consensus (pairs and Cohen's κ) and
   refuses an incomplete CSV.

   ```bash
   uv run python modules/agents_pipeline/scripts/workflow_feasibility.py build-gt
   ```

5. **Run** the 304 runs (needs `ANTHROPIC_API_KEY`):

   ```bash
   uv run python -m gha_agents.experiment modules/agents_pipeline/configs/workflow_feasibility.yaml
   ```

6. **Aggregate and review.** The first pass writes the detection results, one diff per patch, and
   the review sheet `patch_review.csv`. Its `feasibility_category` is `plausible`,
   `plausible_external_validation`, `risky_or_unrelated` or `invalid` on a present pair, and
   `localized_unnecessary` or `behavior_relevant_unnecessary` on an absent one. The second pass
   keeps what the sheet holds, rejects values outside these choices, and writes the refactoring
   summaries.

   ```bash
   uv run python -m utils.workflow_feasibility
   ```

The outputs are in `data/artifacts/workflow_feasibility/`. The two reviewers' sheets, their
adjudication and the findings of a final check are `patch_review_a.csv`, `patch_review_b.csv`,
`patch_review_adjudication.csv` and `patch_review_findings.csv`, in `data/workflow_feasibility/`.
`freeze` records them and the filled `patch_review.csv` in `manifest.json`.

Decisions of the diagnostic:

- **Truth.** The scoring reads `annotations_consensus.csv`, not the label the runs record.
  `uncertain` pairs and missing runs are counted, never scored.
- **One run with a larger cap.** The code sets no output cap, so Agno's default of 8,192 output
  tokens applies, thinking included. D1 on `nedbat@coveragepy@python-nightly.yml`, target
  UnpinnedDependencies, did not fit, and that one run was repeated with 16,384 tokens and a 360 s
  timeout. Its two logs in `runs/output-haiku45/` start with a "One-off exception" line. A run from
  the code may fail on it again.

## Layout

```
configs/                 the experiment YAMLs above
docs/prompts/            the prompts of D1 and D2
docs/workflow_level_feasibility_detection_refactoring_protocol.md
                         the protocol of the feasibility diagnostic
gha_agents/
  experiment.py          YAML -> runs -> dispatch, resume
  cli.py                 gha-smellfix: one design on one workflow
  config.py              the YAML schema
  teams.py               D1Workflow and D2Workflow
  agents.py              output schemas and agent factories
  prompts.py             SMELL_CATALOG, the prompt templates, PROMPT_HASHES
  invoker.py             AgentInvoker, the only call into Agno
  model_factory.py       one model per provider, sampling and thinking
  preflight.py, credit.py
                         the checks before the runs
  record.py              the result JSON and the schema check of the patch
  store.py, trace.py
                         the output layout and the traces
  tools.py               the line and YAML tools the agents call
scripts/
  merge_outputs.py       merges output folders of one model
  workflow_feasibility.py
                         select, consensus, freeze and build-gt
  build_results_csv.py   one CSV row per result
tests/
```

## Checks

```bash
uv run pytest modules/agents_pipeline
```
