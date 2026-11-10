# Refactoring assessment codebook (RQ2)

The rules the annotators applied when they judged the refactoring outputs of RQ2. An output is the
answer of one model, under D1 or D2, for one *(workflow, smell)* pair of the 570-pair set, in the
first repetition.

## How the outputs were judged

- **Annotators.** Two annotators of one pair judged each output: `R1` and `R2`, `R3` and `R4`, or
  `R5` and `R6` in `data/validation/verdicts_raters.csv`. All six judged the 38 outputs of the
  calibration set.
- **Inputs.** The annotators judged each output from the smell's definition and its expected
  constructs (see [`../smell-catalog.md`](../smell-catalog.md)), the model's refactoring
  justification, the diff between the original and the refactored workflow, and the constructs the
  diff changed. There is no reference patch, because a smell can be removed in more than one valid
  way. An "Automatic verdict" line on the screen was a hint only, never the answer.
- **Groups.** Outputs with the same diff, smell, changed constructs and polarity form one group
  (`gkey`). A verdict saved for one output of a group was saved for all of them.
- **Judged outputs.** The annotators judged every output whose model found the smell present, and
  no other. An output whose model found it absent is a detection miss on a positive pair and a
  correct abstention on a negative pair, even when a D1 output changed the workflow. A negative
  output identical to the original needs no judgment either; the tool records the few it showed as
  `preserved`.
- **Uncertain.** When genuinely torn after applying a rule, pick `uncertain`; do not guess.

The annotators do not pick the paper's outcomes. They answer the questions below, and the analysis
computes the outcomes from the answers (see [From answers to outcomes](#from-answers-to-outcomes)).

## A. Positive pair: was the smell removed? (`verdict`: `removed`, `not removed` or `uncertain`)

The pair has the smell. Does the refactored workflow still have it, per the smell's definition?

- **removed**: the refactored workflow no longer exhibits the smell. The mechanism does not have to
  be the expected construct (that is question B); only the smell being gone matters here.
- **not removed**: the smell is still present, OR the output is invalid or malformed (an
  unparseable workflow never removes a smell), OR the change is cosmetic and leaves the smelly
  construct intact.
- **uncertain**: you cannot tell from the diff whether the smell is gone (e.g. it depends on a
  referenced action or secret you cannot see).

Examples:
1. *UnpinnedActionVersion*; the patch changes `uses: actions/checkout@v4` to
   `uses: actions/checkout@<full-sha>`: **removed**.
2. *MissingJobTimeout*; the patch adds a comment but no `timeout-minutes`: **not removed**.
3. *MissingTokenPermissions*; the refactored YAML fails to parse: **not removed** (an invalid
   output cannot have removed the smell).
4. *UnpinnedActionVersion*; the patch pins one of three unpinned actions, and two remain: **not
   removed** (the smell is still present).
5. The smell depends on the behavior of a private composite action that the diff does not show:
   **uncertain**.

## B. Positive pair: was the expected construct changed correctly? (`construct_ok`: `yes`, `no` or `uncertain`)

Independent of A. A model can remove the smell through the wrong construct, or touch the right
construct and still not remove the smell, so answer A and B separately.

- **yes**: the expected construct (or an ancestor or descendant of it) was added or changed, in a
  way that addresses the smell.
- **no**: the expected construct was not touched, or the wrong construct was changed, or it was
  changed incorrectly.
- **uncertain**: you cannot tell whether the touched construct is really the expected one
  (ambiguous abstraction or path).

When the answer is not `yes`, the tool also lists the changed constructs, to mark the unexpected
ones. The marked constructs are in `unexpected`.

Examples:
1. Expected `permissions`; the patch adds top-level `permissions: {contents: read}`: **yes**.
2. Expected `jobs.<job_id>.timeout-minutes`; the patch edits `on.push` instead: **no**.
3. Expected `permissions`; the patch adds `permissions` but leaves it `write-all`: **no**
   (touched, but not correctly).
4. The patch edits a reusable-workflow input whose effect on the expected construct is indirect:
   **uncertain**.

## C. Negative pair: was the change unnecessary? (`verdict`: `unnecessary`, `necessary` or `uncertain`)

The pair does not have the smell, yet the output changed. Should the model have left it alone?

- **unnecessary**: there was no smell, so the change served no purpose toward removing one (even if
  harmless). The model should have made no change.
- **necessary**: the change fixes a real, different problem that genuinely needed fixing (rare for a
  pair without the smell; use it sparingly and explain it in `notes`).
- **uncertain**: you cannot judge whether the change was warranted.

The tool also lists the changed constructs, to mark them. The marked constructs are in
`unexpected`.

Examples:
1. No smell; the patch reformats the YAML or reorders keys, with no semantic effect:
   **unnecessary**.
2. No *MissingJobTimeout* (the timeout is already set); the patch rewrites it to a different valid
   value for no reason: **unnecessary**.
3. No smell, but the patch fixes an unrelated, genuine syntax error of the original: **necessary**
   (note why).

## D. Does the change introduce an evident configuration risk? (`potential_behavior_impact`: `yes`, `no` or `uncertain`)

Asked for every judged output, positive or negative. The tool called this field `risk`; the package
names it after the outcome it feeds, Removal with Potential Behavior-Impacting Change. This is the
subjective security and safety call, so apply the rule strictly. "Evident" means visible in the
diff, not hypothetical. Treat it as binary: a concrete weakening of the workflow's security or
correctness posture, or not.

- **yes**: the change makes the workflow demonstrably less safe or correct: it broadens
  `permissions` (e.g. to `write-all`), un-pins a previously pinned action, adds
  `pull_request_target` with a checkout of untrusted code, injects a secret into an untrusted
  context, removes a timeout, disables a needed check, introduces `${{ }}` injection into a `run:`
  step, etc.
- **no**: no such weakening is evident; the change is neutral or improves the posture.
- **uncertain**: you suspect a risk but cannot confirm it from the diff alone.

Examples:
1. The patch changes `permissions: read-all` to `permissions: write-all`: **yes**.
2. The patch pins an action to a full SHA: **no** (it improves the posture).
3. The patch adds `timeout-minutes: 10` where there was none: **no**.
4. The patch interpolates `${{ github.event.issue.title }}` directly into a shell `run:` line:
   **yes** (a script-injection sink).
5. The patch adds an `env:` value whose source you cannot see: **uncertain**.

## E. Negative pair: what kind of edit is it? (`edit_type`: `cosmetic edits`, `localized but unnecessary edits` or `behavior-relevant edits`)

Asked only for negative outputs that changed.

- **cosmetic edits**: formatting, comments, names. Usually benign, but unnecessary.
- **localized but unnecessary edits**: an added timeout, changed permissions, an added step
  condition. It looks like a plausible fix, but the target smell was absent.
- **behavior-relevant edits**: changes to triggers, `runs-on`, `needs`, `run`, `uses`, the matrix or
  the permissions. Potentially risky, because they may alter the workflow's behavior with no
  smell-removal benefit.

Use the most specific label that matches the diff. If an edit clearly affects behavior, prefer
**behavior-relevant edits**, even if it is also localized. If it only changes formatting or names,
use **cosmetic edits**.

## Disagreements

- A disagreement between the two annotators on `verdict`, on `construct_ok` (positive pairs) or on
  `potential_behavior_impact` went to an adjudicator, whose choice is final
  (`data/validation/disagreement_resolutions.csv`). Disagreements on `edit_type` were not
  adjudicated.
- `data/validation/manual_validation_final.csv` holds the resolved values of every judged output;
  `resolution_source` says whether both annotators agreed or the adjudicator decided.
- Notebook `05_validation` reports Cohen's κ before adjudication (`utils/inter_rater.py`). The target
  was κ ≥ 0.6.

## From answers to outcomes

The analysis counts each output on its own and adds the automatic schema check (`patch_valid`); the
[analysis README](../../modules/analysis/README.md#decisions-the-paper-does-not-state) lists these
rules.

Positive pairs (`tab:rq2_refactoring_outcomes_by_smell`, `tab:rq2_end_to_end_conditional`,
`tab:rq2_clean_removal_by_model`):

| Outcome | Condition |
|---|---|
| Clean Removal (CR) | `verdict` `removed`, `construct_ok` `yes`, `potential_behavior_impact` `no`, and the patch passes the schema check |
| Removal with Potential Behavior-Impacting Change (BR) | `verdict` `removed` and `potential_behavior_impact` `yes` |
| Removal with Unexpected Changes (UX) | `verdict` `removed`, `potential_behavior_impact` `no`, and `construct_ok` `no` or a patch that fails the schema check |
| Failed or No Patch (FN) | `verdict` `not removed`, or a detection miss |

An output with `uncertain` in `verdict`, `construct_ok` or `potential_behavior_impact` is not
assessable and is left out. The smell removal rate (SRR) is CR + UX + BR over the assessable
outputs.

Negative pairs:

| Outcome | Condition |
|---|---|
| Correct Abstention (CA) | the model found the smell absent, or the output is unchanged, or its `verdict` is `preserved` or `necessary` |
| Unnecessary Edit (UE) | `verdict` `unnecessary` |

An output whose `verdict` is `uncertain` is left out.

The edit types of `tab:rq2_negative_edit_types` do not come from `edit_type`: the analysis classifies
each unnecessary edit by the riskiest construct it changes. A `timeout-minutes` change is localized,
a naming change is cosmetic, and any other change is behavior-relevant. `edit_type` enters only
Cohen's κ, in place of `construct_ok` on negative pairs. `unexpected` and `notes` enter no number.
