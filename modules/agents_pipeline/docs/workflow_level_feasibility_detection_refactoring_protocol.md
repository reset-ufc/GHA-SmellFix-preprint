# Workflow-Level Feasibility Diagnostic Protocol  
## Detection and Refactoring in Multi-Smell Workflows

## Purpose

This protocol defines a small **workflow-level feasibility diagnostic** for assessing whether the current target-conditioned design of **the pipeline** can be operationalized on workflows that contain multiple smells.

The diagnostic evaluates two aspects:

```text
1. Detection feasibility:
   Can the design recover manually annotated smells when applied exhaustively to a workflow?

2. Refactoring feasibility:
   When a smell is detected, can the design generate a plausible smell-specific candidate patch?
```

The diagnostic is intentionally limited. It is **not** an open-ended discovery evaluation, not a benchmark, not a prevalence study, and not a coordinated multi-smell repair evaluation.

---

## Scope

The pipeline remains **target-conditioned**. Each run receives:

```text
workflow
repository context
one target smell definition
```

Therefore, this diagnostic evaluates:

```text
exhaustive target-conditioned screening and smell-specific refactoring
in workflows that contain multiple smells
```

It does **not** evaluate:

```text
open-ended multi-smell discovery
coordinated multi-smell repair
safe merging of multiple patches
workflow-level smell prevalence
general model superiority
benchmark-level performance
```

Generated patches are assessed **independently** for each target smell. Patches generated for different target smells in the same workflow must not be merged into a single workflow-level repair.

---

## Study Design

Use **8 workflows**, **19 smells**, and **2 orchestration designs**.

```text
8 workflows × 19 smells × 2 designs = 304 target-conditioned runs
```

Designs:

```text
D1: joint detect-and-refactor
D2: detect-then-refactor
```

Model:

```text
Claude Haiku 4.5
```

Execution setup:

```text
same temperature, prompt templates, schemas, and execution protocol used in the main study
```

---

## Workflow Selection

Select **8 real workflows** from the mined corpus, preferably outside the main 570-pair evaluation set.

### Inclusion Criteria

Each selected workflow should:

```text
- satisfy the same inclusion criteria used in the main study;
- be valid YAML;
- be structurally valid as a GitHub Actions workflow;
- contain evidence of at least two distinct smells;
- include smells that are refactorable through concrete workflow edits;
- include both localized and context-dependent smells when possible;
- be small enough for manual annotation and patch inspection.
```

### Suggested Distribution

```text
2 workflows with 2 annotated smells
3 workflows with 3–4 annotated smells
3 workflows with 5+ annotated smells, if available
```

The GHA analyzer may be used to guide workflow selection, but it must not be treated as ground truth.

---

## Manual Workflow-Level Annotation

For each of the 8 workflows, manually annotate all 19 smells.

### Annotation Unit

```text
(workflow, smell)
```

### Annotation Fields

```text
workflow_id
repository
workflow_path
smell_name
manual_label: present / absent / uncertain
number_of_instances
smell_evidence
expected_refactoring_scope
notes
```

### Annotation Rules

```text
present   = at least one instance of the smell exists in the workflow
absent    = no instance of the smell is found
uncertain = static evidence is insufficient for a reliable decision
```

Two authors should annotate the workflows and resolve disagreements by consensus.

Because this is a small feasibility diagnostic, inter-rater agreement statistics are optional. Uncertain cases should be excluded from the main descriptive metrics, but their number should be reported.

---

## Execution Procedure

For each selected workflow:

```text
for design in {D1, D2}:
    for smell in 19 smells:
        run the pipeline with:
            workflow
            repository context
            target smell definition
            selected design
            Claude Haiku 4.5
        store:
            detection verdict
            generated patch, if any
            detection/refactoring rationale
            execution metadata
```

This produces:

```text
152 runs for D1
152 runs for D2
304 runs total
```

---

## Stored Output per Run

For each `(workflow, smell, design)` run, store:

```text
workflow_id
repository
workflow_path
design
target_smell
manual_label
detection_verdict
patch_generated: yes / no
original_workflow
candidate_workflow, if any
patch_diff, if any
rationale
latency
input_tokens
output_tokens
estimated_cost
```

---

# Part 1 — Detection Feasibility

## Detection Unit

The detection unit is:

```text
(workflow, smell, design)
```

The detection question is:

```text
When the current target-conditioned design is applied once per smell,
can it recover the manually annotated smell set of a multi-smell workflow?
```

---

## Detection Aggregation

### Pair Level

For each `(workflow, smell, design)`:

```text
True Positive  = manual_label present and predicted present
False Negative = manual_label present and predicted absent
False Positive = manual_label absent and predicted present
True Negative  = manual_label absent and predicted absent
```

Exclude uncertain manual labels from the main metrics.

### Workflow Level

For each `(workflow, design)`:

```text
annotated_present_smells = set of manually annotated present smells
predicted_present_smells = set of predicted-present smells

missed_smells = annotated_present_smells - predicted_present_smells
extra_smells  = predicted_present_smells - annotated_present_smells
```

---

## Detection Metrics

Report only descriptive metrics.

### Pair-Level Detection Metrics

```text
workflow-smell precision = TP / (TP + FP)
workflow-smell recall    = TP / (TP + FN)
workflow-smell F1        = optional
```

### Workflow-Level Detection Metrics

```text
exact set match = number of workflows where predicted set equals annotated set
partial match   = number of workflows where at least one annotated smell is recovered
mean missed smells per workflow
mean extra smells per workflow
```

### D1 vs D2 Detection Summary

Report descriptively:

```text
D1 recovered A of B manually annotated smell instances.
D2 recovered C of B manually annotated smell instances.

D1 suggested D additional smell instances.
D2 suggested E additional smell instances.

D1 exact-match workflows: F / 8
D2 exact-match workflows: G / 8
```

Do not perform statistical tests.

---

# Part 2 — Refactoring Feasibility

## Refactoring Unit

The refactoring unit is:

```text
(workflow, target smell, design)
```

The refactoring question is:

```text
When the target smell is manually present and the design attempts a repair,
can it generate a plausible smell-specific candidate patch?
```

Detection and refactoring are connected because the current designs include detection as part of the pipeline. Therefore, report when no patch is generated because the smell was missed.

---

## Patch Assessment Scope

Each generated patch is assessed independently for its target smell.

```text
one target-smell run → one candidate patch → one independent assessment
```

Do **not** merge patches from multiple target-smell runs.

Do **not** evaluate coordinated multi-smell repair.

Do **not** evaluate whether multiple generated patches can be safely composed into a single workflow-level patch.

---

## Automatic Patch Checks

Apply lightweight automatic checks to each generated patch:

```text
- candidate output exists;
- output can be parsed as YAML;
- output is structurally valid as a GitHub Actions workflow;
- output differs from the original workflow;
- a diff can be computed between original and candidate workflow.
```

These checks support feasibility classification; they do not constitute a full refactoring evaluation.

---

## Lightweight Refactoring Feasibility Categories

For each generated patch, classify it into one of five descriptive categories.

```text
Plausible smell-specific refactoring
Plausible but requires external validation
Risky or unrelated refactoring
Invalid or unusable patch
No patch due to missed detection/abstention
```

### 1. Plausible smell-specific refactoring

Use when the patch appears to address the target smell and stays within the expected smell-related construct.

Examples:

```text
- adds timeout-minutes for MissingJobTimeout;
- adds or scopes permissions for MissingTokenPermissions or MissingPermExternalActions;
- adds concurrency for MissingConcurrencyPR or MissingConcurrencyBranch;
- adds path filters for MissingPathFilters;
- adds a fork guard for MissingForkGuardPR or UnsafeDeployOnFork;
- introduces a matrix for SingleOSExecution or SingleLangVersionCI.
```

### 2. Plausible but requires external validation

Use when the patch targets the smell but depends on facts outside the YAML.

Examples:

```text
- replaces a mutable action reference with a commit-like reference that must be checked;
- pins a dependency version that may not exist;
- changes a reusable workflow reference that must be validated in the repository;
- adds a condition whose correctness depends on repository-specific policy;
- changes cache parameters that require repository-specific validation.
```

### 3. Risky or unrelated refactoring

Use when the patch changes constructs unrelated to the target smell or may affect workflow behavior beyond the intended smell-specific repair.

Examples:

```text
- changes triggers unrelated to the target smell;
- changes runner labels or matrix values unnecessarily;
- changes dependency or language versions unrelated to the target smell;
- rewrites reusable workflow calls;
- modifies deployment, release, artifact, or secret-related behavior without clear need.
```

### 4. Invalid or unusable patch

Use when the output is malformed or cannot be used as a candidate patch.

Examples:

```text
- invalid YAML;
- invalid GitHub Actions structure;
- incomplete workflow;
- natural-language output instead of YAML;
- empty output;
- patch does not address the target smell at all.
```

### 5. No patch due to missed detection/abstention

Use when the smell is manually annotated as present, but the design predicts it as absent or returns the original workflow unchanged.

This category distinguishes refactoring failure caused by missed detection from failure caused by patch generation.

---

## Handling Annotated-Absent Smells

For manually absent smells, the expected behavior is abstention.

If a patch is generated for an annotated-absent smell, classify it as:

```text
unnecessary patch generated
```

Optionally sub-classify it as:

```text
localized but unnecessary
behavior-relevant unnecessary
```

This is secondary. It is reported only to indicate whether applying the design exhaustively creates obvious over-refactoring.

---

## Refactoring Metrics

Report descriptive metrics only.

### For Manually Present Smells

For each design:

```text
manually present smell runs
predicted-present smell runs
patches generated
plausible smell-specific refactorings
plausible but requiring external validation
risky or unrelated refactorings
invalid or unusable patches
no patch due to missed detection/abstention
```

Optional rates:

```text
patch generation rate =
patches generated / manually present smell runs

plausible refactoring rate =
(plausible smell-specific refactorings + plausible but requiring external validation)
/
manually present smell runs

conditional plausible refactoring rate =
(plausible smell-specific refactorings + plausible but requiring external validation)
/
patches generated
```

### For Manually Absent Smells

For each design:

```text
manually absent smell runs
correct abstentions
unnecessary patches generated
behavior-relevant unnecessary patches, optional
```

Optional rate:

```text
unnecessary patch rate =
unnecessary patches generated / manually absent smell runs
```

---

## Multi-Smell Interaction Notes

If a patch for one target smell modifies constructs related to another smell, record it qualitatively:

```text
target_smell
other_smell_construct_modified: yes / no
other_smell_name, if identifiable
effect: likely improves / likely worsens / unclear
notes
```

Do not count this as coordinated multi-smell repair.

Interpretation:

```text
- if the edit is outside the target-smell scope and harmless, classify as plausible but note collateral improvement;
- if the edit is outside the target-smell scope and risky, classify as risky or unrelated;
- if the effect cannot be determined, classify as requiring external validation or unclear.
```

---

## Manual Review Form

Use one row per generated output.

```text
workflow_id:
repository:
workflow_path:
design: D1 / D2
target_smell:
manual_label: present / absent / uncertain
detection_verdict: present / absent
patch_generated: yes / no
yaml_valid: yes / no / not_applicable
gha_structurally_valid: yes / no / not_applicable
target_smell_appears_addressed: yes / no / unclear / not_applicable
modifies_expected_construct: yes / no / partial / unclear
modifies_other_smell_construct: yes / no / unclear
requires_external_validation: yes / no
potential_behavior_impact: yes / no / unclear
feasibility_category:
brief_justification:
```

---

# Combined Reporting

## D1 vs D2 Summary

Because the same workflows and smells are used for both designs, summarize differences descriptively.

```text
Detection:
  D1 recovered A of B manually annotated smell instances.
  D2 recovered C of B manually annotated smell instances.
  D1 suggested D additional smell instances.
  D2 suggested E additional smell instances.

Refactoring:
  D1 generated F patches for manually present smells:
      G plausible smell-specific
      H plausible but requiring external validation
      I risky or unrelated
      J invalid or unusable
      K no patch due to missed detection/abstention

  D2 generated L patches for manually present smells:
      M plausible smell-specific
      N plausible but requiring external validation
      O risky or unrelated
      P invalid or unusable
      Q no patch due to missed detection/abstention

Abstention:
  D1 generated R unnecessary patches for manually absent smells.
  D2 generated S unnecessary patches for manually absent smells.
```

Avoid claiming superiority. The diagnostic is too small for that.

---

## Reporting Template for the Paper

```latex
\textbf{Workflow-level feasibility diagnostic.}
Because our approach is target-conditioned, our main evaluation does not assess open-ended multi-smell discovery or coordinated multi-smell repair. To examine whether the same design can be operationalized on workflows containing multiple smells, we conducted a small feasibility diagnostic on eight workflows manually annotated for all 19 smells. For each workflow, we ran both D1 and D2 once per target smell, yielding 304 target-conditioned runs. We aggregated predicted-present smells to assess workflow-level detection feasibility and inspected generated patches to assess smell-specific refactoring feasibility. Generated patches were assessed independently for each target smell and were not merged into workflow-level repairs.
```

### Result Paragraph Template

```latex
Across eight workflows and 304 target-conditioned runs, D1 recovered X of Y manually annotated smell instances, while D2 recovered A of Y. D1 suggested Z additional smell instances and D2 suggested W. For refactoring, D1 generated P patches for manually present smells, of which Q were plausible smell-specific refactorings and R required external validation; D2 generated S patches, of which T were plausible smell-specific refactorings and U required external validation. The remaining outputs were risky/unrelated, invalid/unusable, unnecessary for manually absent smells, or produced no patch due to missed detection or abstention. These results provide feasibility evidence that the current target-conditioned design can be applied to multi-smell workflows for both detection and smell-specific refactoring, but they should not be interpreted as open-ended discovery, coordinated multi-smell repair, or a general performance estimate.
```

### Scope Clarification

```latex
This diagnostic should not be interpreted as open-ended discovery or coordinated multi-smell repair. It evaluates whether the same target-conditioned design can be applied to workflows with multiple smells by running one smell-specific task per target smell. Generated patches are assessed independently; we do not evaluate whether multiple patches can be safely merged into a single workflow-level repair.
```

---

## What Not to Claim

Do not claim that this diagnostic evaluates:

```text
open-ended discovery
general workflow-level performance
natural smell prevalence
model superiority
complete multi-smell repair
coordinated patch composition
safe patch merging
benchmark-level evidence
```

Preferred wording:

```text
workflow-level feasibility diagnostic
detection feasibility
smell-specific refactoring feasibility
multi-smell workflow context
descriptive evidence
```

Avoid wording such as:

```text
benchmark
open discovery
exhaustive evaluation
coordinated repair
generalizable workflow-level performance
```

---

## Checklist

```text
[ ] Select 8 real workflows with multiple smells.
[ ] Ensure workflows satisfy study inclusion criteria.
[ ] Manually annotate all 19 smells in each workflow.
[ ] Record expected refactoring scope for present smells.
[ ] Resolve annotation disagreements by consensus.
[ ] Run Claude Haiku 4.5 with D1 for 8 × 19 = 152 runs.
[ ] Run Claude Haiku 4.5 with D2 for 8 × 19 = 152 runs.
[ ] Store detection verdicts, generated patches, diffs, rationales, and metadata.
[ ] Aggregate predicted-present smells by workflow and design.
[ ] Compute descriptive detection metrics.
[ ] Assess generated patches independently for each target smell.
[ ] Do not merge patches across target smells.
[ ] Classify refactoring outputs using lightweight feasibility categories.
[ ] Report descriptive detection and refactoring evidence.
[ ] Avoid statistical tests.
[ ] Avoid terms such as benchmark, open discovery, exhaustive evaluation, coordinated repair, or prevalence estimate.
```
