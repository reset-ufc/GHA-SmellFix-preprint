"""prompts.py — Smell catalog, system prompt templates, and builders.

A ``Scenario`` is one of the paper's two designs: D1, the joint design, and D2, the
decomposed design (see ``gha_agents.config``).
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .output_contract import OUTPUT_CONTRACTS
from .user_turn import all_user_turn_tags

__all__ = [
    "D1_COMBINED_TEMPLATE",
    "D2_DETECTION_TEMPLATE",
    "D2_REFACTOR_TEMPLATE",
    "PROMPT_HASHES",
    "SMELL_CATALOG",
    "Role",
    "Scenario",
    "build_combined_prompt",
    "build_detection_prompt",
    "build_refactor_prompt",
    "get_all_prompt_hashes",
    "prompt_hash",
]

Scenario = Literal["D1", "D2"]
Role = Literal[
    "combined",
    "detection",
    "detection_validation",
    "refactor",
    "refactor_validation",
]

# ─── Smell Catalog ──────────────────────────────────────────────────
# ``pascal`` is the paper's short identifier (tables/smell-catalog.tex). Each
# description is the v4 definition. Only the 19 behavioral smells evaluated in
# the paper are listed; the 3 mechanically-detectable presentational smells
# (Add comments / Name run steps / Correct indentation) are excluded — they need
# no LLM judgment and are not part of the evaluation.
SMELL_CATALOG = [
    # ── Security (IDs 1-3) ───────────────────────────
    {
        "name": "Define permissions for workflows with external actions",
        "pascal": "MissingPermExternalActions",
        "description": """A job exhibits this smell only if ALL conditions are true: (1) it contains at least one step-level `uses:` on a `job`; (2) at least one reference is a GitHub action in the form `owner/repo@ref` or `owner/repo/path@ref`; (3) the `owner/repo` portion, differs from the provided `current_repository`; and (4) the job has no effective `permissions:` configuration, meaning there is neither a workflow-level nor job-level `permissions:` block. Treat relative-path references, such as `uses: ./.github/actions/setup`, as local actions. Do not flag local actions, same-repository action references, job-level `uses:` references to reusable workflows, Docker image references such as `uses: docker://alpine:3.19`, unparsable references, or jobs with any applicable `permissions:` block, even if permissions are broad. Flag one occurrence per job, regardless of how many external step-level actions it contains.""",
    },
    {
        "name": "Use commit hash instead of tags for action versions",
        "pascal": "UnpinnedActionVersion",
        "description": """A step-level `uses:` reference exhibits this smell only if ALL conditions are true: (1) the reference appears under `jobs.<job_id>.steps[*].uses`; (2) the `uses:` value is a GitHub action reference in the form `owner/repo@ref` or `owner/repo/path@ref`; and (3) `ref` is not exactly a 40-character hexadecimal commit SHA. Flag branch names, tags, major-version aliases, and floating references such as `@main`, `@master`, `@develop`, `@v1`, `@v2`, `@latest`, or any other non-40-character hexadecimal reference. This applies to external actions and to same-repository actions when they are referenced through the explicit `owner/repo@ref` or `owner/repo/path@ref` form. Do not flag local actions referenced by relative path, such as `uses: ./.github/actions/setup`, because they do not use an explicit `@ref`. Do not flag actions pinned to a full 40-character hexadecimal commit SHA, `uses:` values without `@ref`, non-GitHub Docker image references such as `uses: docker://alpine:3.19`, job-level `uses:` references to reusable workflows, including local workflows such as `jobs.<job_id>.uses: ./.github/workflows/ci.yml`, or references whose value cannot be parsed confidently. Flag one occurrence for each step-level `uses:` reference that satisfies all conditions.""",
    },
    {
        "name": "Set permissions for GitHub Token",
        "pascal": "MissingTokenPermissions",
        "description": """A job exhibits this smell only if ALL conditions are true: (1) the job uses the GitHub-provided token explicitly through `secrets.GITHUB_TOKEN` or `github.token`, or uses an action/command that clearly performs authenticated GitHub operations such as creating comments, labels, releases, deployments, packages, statuses, pull request updates, issue updates, or repository writes; (2) the job has no effective `permissions:` configuration, either from a workflow-level `permissions:` block or a job-level `permissions:` block. Do not flag jobs with an applicable workflow-level or job-level `permissions:` block, even if the permissions are broad. Do not flag jobs only because they run on GitHub Actions, use `actions/checkout`, or perform ordinary build/test steps. This smell concerns only the absence of explicit token permission configuration, not whether the configured permissions follow least privilege. When token use is uncertain, do not flag.""",
    },
    # ── Performance / Optimization (IDs 4-13) ─────────
    {
        "name": "Prevent running issue/PR actions on forks",
        "pascal": "MissingForkGuardPR",
        "description": """A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by an issue- or pull-request-related event that may involve forked repositories, such as `pull_request`, `pull_request_target`, `issue_comment`, `pull_request_review`, or `pull_request_review_comment`; (2) at least one job or relevant step performs an issue or pull request action, such as commenting, labeling, approving, updating statuses, or modifying issue/PR metadata; and (3) that job or step has no effective fork guard. Effective fork guards must clearly prevent the issue/PR action from running for forked pull requests, such as `github.event.pull_request.head.repo.fork == false`, `github.event.pull_request.head.repo.full_name == github.repository`, or equivalent same-repository PR checks. For `issue_comment`, the guard must also ensure the comment belongs to a pull request before checking the associated PR fork status. Do not treat repository-identity checks alone, label-based checks, branch checks, actor checks, event-type checks, or comment-content checks as fork guards unless combined with an explicit fork-status or same-repository PR check. Do not flag workflows that are not triggered by issue/PR-related events, do not perform issue/PR actions, are protected by effective fork guards, or cannot be determined confidently. Flag one occurrence for each workflow that satisfies all conditions.""",
    },
    {
        "name": "Avoid jobs without timeouts",
        "pascal": "MissingJobTimeout",
        "description": """A job exhibits this smell only if ALL conditions are true: (1) it is a job entry defined in the current workflow file; and (2) it does not define `timeout-minutes` at the job level. Flag one occurrence for each job that satisfies these conditions, even if one or more steps define their own `timeout-minutes`, and even if the job has no `steps` block. Step-level `timeout-minutes` does not count as a job-level timeout and does not prevent this smell. Do not flag jobs that explicitly define `timeout-minutes` at the job level or jobs whose structure or timeout configuration cannot be determined confidently. This rule checks only whether `timeout-minutes` is present at the job level in the current workflow file; it does not inspect called reusable workflows or their internal timeout configuration.""",
    },
    {
        "name": "Stop running workflows when there is a newer commit in PR",
        "pascal": "MissingConcurrencyPR",
        "description": """A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `pull_request` or `pull_request_target`; (2) it does not define an effective workflow-level `concurrency:` configuration with `cancel-in-progress: true`; and (3) at least one job that can run on the PR trigger does not define an effective job-level `concurrency:` configuration with `cancel-in-progress: true`. Flag one occurrence for each workflow where newer commits to a pull request may start additional workflow runs without canceling older in-progress runs for that PR. Do not flag workflows without `pull_request` or `pull_request_target` triggers. Do not flag workflows with workflow-level `concurrency.cancel-in-progress: true`, or workflows where every job that can run on the PR trigger has job-level `concurrency.cancel-in-progress: true`. For this smell, only check whether an applicable `concurrency:` block with `cancel-in-progress: true` exists. Do not evaluate whether the `concurrency.group` expression is optimal, pull-request-specific, branch-specific, ref-specific, workflow-specific, or otherwise well designed. Do not flag configurations whose `concurrency:` or `cancel-in-progress` value cannot be confidently interpreted. When uncertain, do not flag.""",
    },
    {
        "name": "Stop running workflows when there is a newer commit in branch",
        "pascal": "MissingConcurrencyBranch",
        "description": """A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `push` events; (2) it does not define an effective workflow-level `concurrency:` configuration with `cancel-in-progress: true`; and (3) at least one job that can run on branch `push` events does not define an effective job-level `concurrency:` configuration with `cancel-in-progress: true`. Flag one occurrence for each workflow where newer commits pushed to a branch may start additional workflow runs without canceling older in-progress runs for that branch. Do not flag workflows that are not triggered by `push`, workflows triggered only by tag pushes, workflows with workflow-level `concurrency.cancel-in-progress: true`, or workflows where every job that can run on branch `push` events has job-level `concurrency.cancel-in-progress: true`. For this smell, only check whether an applicable `concurrency:` block with `cancel-in-progress: true` exists. Do not evaluate whether the `concurrency.group` expression is optimal, branch-specific, workflow-specific, or otherwise well designed. Do not flag configurations whose `concurrency:` or `cancel-in-progress` value cannot be confidently interpreted. When uncertain, do not flag.""",
    },
    {
        "name": "Avoid running CI actions when no source code has changed",
        "pascal": "MissingPathFilters",
        "description": """A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `push` or `pull_request`; (2) its main purpose is CI validation, such as build, test, lint, static analysis, packaging checks, or dependency checks; and (3) at least one relevant `push` or `pull_request` trigger does not define `paths` or `paths-ignore`. Flag one occurrence for each workflow whose relevant CI trigger lacks `paths` or `paths-ignore`. Do not flag workflows where every relevant `push` or `pull_request` trigger defines `paths` or `paths-ignore`, regardless of whether the filter is complete, precise, or optimal. Do not flag workflows whose main purpose is documentation, release, deployment, notification, scheduled maintenance, formatting-only automation, whole-repository automation, or cases where the workflow purpose cannot be determined confidently. When the trigger, workflow purpose, or path-filter configuration cannot be determined confidently, do not flag.""",
    },
    {
        "name": "Avoid executing scheduled workflows on forks",
        "pascal": "MissingForkGuardSchedule",
        "description": """A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `schedule`; (2) at least one scheduled job can run without a condition restricting execution to the canonical repository or repository owner; and (3) there is no effective workflow-level or job-level `if:` condition using values such as `github.repository`, `github.repository_owner` or `github.event.repository.fork` to restrict scheduled execution. Do not flag workflows that are not triggered by `schedule`, or scheduled workflows where every scheduled job is protected by a repository or owner check, such as `github.repository == 'owner/repo'`, `github.repository_owner == 'owner'` or `github.event.repository.fork == false'`. When uncertain whether the guard applies to every scheduled job, do not flag.""",
    },
    {
        "name": "Avoid uploading artifacts on forks",
        "pascal": "UnsafeArtifactUploadFork",
        "description": """An artifact-upload step exhibits this smell only if ALL conditions are true: (1) the enclosing workflow is triggered by pull_request or pull_request_target; (2) the step uploads retained GitHub Actions workflow artifacts, i.e., artifacts stored by GitHub Actions as part of the workflow run, using actions/upload-artifact, actions/upload-pages-artifact, or a wrapper explicitly compatible with actions/upload-artifact; and (3) neither the artifact-upload step nor its enclosing job has an effective condition preventing execution for forked pull requests. Effective fork guards include conditions such as github.event.pull_request.head.repo.fork == false, github.event.repository.fork == false, or equivalent checks that clearly prevent artifact upload when the workflow is running for a fork. Do not flag artifact-upload steps in workflows that are not triggered by pull_request or pull_request_target. Do not flag coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or files uploaded to external services, registries, cloud storage, or releases, because these are not GitHub Actions workflow artifacts. Do not flag steps that only move, copy, print, cache, download, or pass files within the same runner, job, or workflow without storing them as retained GitHub Actions workflow artifacts. Do not flag artifact-upload steps protected by an effective fork-status guard or cases where the trigger, artifact-upload behavior, or fork guard cannot be determined confidently. Flag one occurrence for each artifact-upload step that satisfies all conditions.""",
    },
    {
        "name": "Use 'if' for upload-artifact action",
        "pascal": "MissingUploadCondition",
        "description": """A step exhibits this smell only if ALL conditions are true: (1) it uploads retained GitHub Actions workflow artifacts using `actions/upload-artifact`, `actions/upload-pages-artifact`, or a wrapper explicitly compatible with `actions/upload-artifact`; (2) the artifact-upload step does not define an explicit step-level `if:` condition; and (3) the enclosing job does not define an explicit job-level `if:` condition. Do not flag artifact-upload steps that define their own step-level `if:` condition, artifact-upload steps inside jobs with a job-level `if:` condition, steps that do not create retained GitHub Actions workflow artifacts, or steps whose upload behavior cannot be determined confidently. Do not treat coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or uploads to external services, registries, cloud storage, or GitHub Releases as retained GitHub Actions workflow artifact uploads. This smell concerns only the absence of an explicit condition controlling artifact upload, not whether the condition is sufficiently restrictive. Flag one occurrence for each artifact-upload step that satisfies all conditions.""",
    },
    {
        "name": "Avoid deploying jobs on forks",
        "pascal": "UnsafeDeployOnFork",
        "description": """A job exhibits this smell only if ALL conditions are true: (1) the workflow can run in a fork-related pull request context, such as `pull_request`, `pull_request_target`, or another PR-based event; (2) the job directly performs an external delivery operation in the current workflow file through a step-level `uses:` action or a `run:` command, such as deployment, release creation, package publishing, container image push, or cloud/provider deployment; and (3) neither the job nor the delivery step has an effective fork guard. Do not treat job names, workflow names, comments, or job-level reusable workflow calls through ⁠ jobs.<job_id>.uses ⁠ as sufficient evidence of external delivery by themselves. Concrete evidence of external delivery must be visible in the current workflow file, such as `npm publish`, `docker push`, registry login followed by push, cloud deployment commands, release creation actions, deployment actions, deployment environments, or secrets clearly used for delivery. Do not flag reusable-workflow-only jobs. Do not flag ordinary build, test, lint, analysis, packaging-only, artifact-upload-only, image-build-only, ambiguous jobs, workflows that cannot run from fork-related PR contexts, or jobs protected by an effective fork guard. When uncertain, do not flag. Effective fork guards include conditions such as `github.event.pull_request.head.repo.fork == false`, `github.event.pull_request.head.repo.full_name == github.repository`, `github.event.repository.fork == false`, or equivalent fork-status or same-repository PR checks. Flag one occurrence for each job that satisfies all conditions.""",
    },
    {
        "name": "Avoid starting new workflow while previous one is running",
        "pascal": "MissingConcurrencyGlobal",
        "description": """A workflow or job exhibits this smell only if ALL conditions are true: (1) it can be triggered repeatedly for the same logical task, such as scheduled automation, deployment, release, publishing, synchronization, cleanup, data processing, or other long-running or side-effectful automation; (2) overlapping runs could cause unnecessary resource consumption, duplicate side effects, conflicting updates, repeated publishing/deployment, or resource contention; and (3) there is no effective workflow-level or job-level `concurrency:` group that prevents simultaneous or overlapping runs for that same task. Do not flag ordinary CI, build, lint, or test workflows unless there is explicit evidence that overlapping runs are harmful, long-running, resource-intensive, or side-effectful. Do not flag workflows or jobs that already define an effective `concurrency:` group for the same task, with or without `cancel-in-progress`, if it prevents simultaneous overlapping execution. When the repeated trigger, overlap risk, or concurrency configuration cannot be determined confidently, do not flag.""",
    },
    # ── Other CI/CD (IDs 14-22) ───────────────────
    {
        "name": "Use fixed version for runs-on argument",
        "pascal": "MutableRunnerVersion",
        "description": """A job exhibits this smell only if ALL conditions are true: (1) the job defines `runs-on` directly in the current workflow file; and (2) the effective `runs-on` value includes a mutable GitHub-hosted runner label ending in `-latest`, such as `ubuntu-latest`, `windows-latest`, or `macos-latest`. Flag one occurrence for each job that satisfies these conditions. If `runs-on` is a list, flag the job if any label includes a mutable GitHub-hosted runner label ending in `-latest`. Do not flag jobs that use only fixed runner labels such as `ubuntu-22.04`, `windows-2022`, or `macos-14`, self-hosted runner labels, reusable workflow calls using `jobs.<job_id>.uses`, missing `runs-on`, or parameterized `runs-on` values that cannot be resolved from the same file. When the runner label cannot be determined confidently, do not flag.""",
    },
    {
        "name": "Use cache parameter instead of cache option",
        "pascal": "SuboptimalCacheUsage",
        "description": """A dependency-cache step exhibits this smell only if ALL conditions are true: (1) the step uses `actions/cache`; (2) the enclosing workflow uses a language or runtime setup action with documented built-in dependency caching, such as `actions/setup-node`, `actions/setup-java`, `actions/setup-python`, `actions/setup-go`, or `ruby/setup-ruby`; (3) the `actions/cache` step caches dependencies for the same language, runtime, or package manager handled by that setup action; and (4) the cache target is not clearly used for another purpose, such as build outputs, generated artifacts, compiler caches, Docker layers, tool downloads, or cross-job cache sharing. Flag one occurrence for each dependency-cache step that satisfies these conditions. Do not flag cache steps for unrelated ecosystems, build outputs, generated artifacts, compiler caches, Docker layers, tool downloads, cross-job cache sharing, or setup actions without a clear built-in dependency caching option. Do not flag cases where the setup action, cache target, or ecosystem match cannot be determined confidently. When uncertain, do not flag.""",
    },
    {
        "name": "Use single-command steps",
        "pascal": "MultiCommandStep",
        "description": """A `run:` step exhibits this smell only if ALL conditions are true: (1) the step contains two or more relevant shell commands in the same `run:` block; (2) at least two of these commands belong to different workflow phases, such as install, build, test, lint, package, deploy, upload, or cleanup; and (3) these phase-related commands could reasonably be executed as separate `run:` steps. Relevant commands are commands that perform meaningful workflow operations. Do not count auxiliary shell constructs or helper commands by themselves, such as `echo`, `cd`, `pwd`, `ls`, `mkdir`, `export`, variable assignments, conditionals, loops, comments, `true`, `false`, or `sleep`, unless they clearly implement a distinct workflow phase. Flag one occurrence for each `run:` step that satisfies these conditions. Do not flag single relevant commands, commands split across lines only for readability, script invocations such as `npm test` or `python script.py`, short condition/fallback commands for one operation, tightly coupled commands that perform one task, or cases where the commands, phases, or separability cannot be determined confidently.""",
    },
    {
        "name": "Run tests on multiple OS's",
        "pascal": "SingleOSExecution",
        "description": """A job exhibits this smell only if ALL conditions are true: (1) the job is a test job, identified by test-related job names or commands such as `test`, `pytest`, `npm test`, `yarn test`, `mvn test`, `gradle test`, `go test`, `cargo test`, or equivalent; (2) the job has exactly one effective operating system in its `runs-on` configuration, or equivalent runner configuration; and (3) the job does not define an OS matrix or equivalent strategy with two or more distinct operating system values. Flag one occurrence for each test job that satisfies these conditions. If the job defines a matrix, but the matrix contains only one distinct operating system value, still flag the job because it effectively tests on only one OS. Do not flag non-test jobs, deployment or documentation jobs, jobs that effectively test on two or more operating systems, jobs running inside a fixed container environment, or cases where the job purpose or operating system configuration cannot be determined confidently. When uncertain, do not flag.""",
    },
    {
        "name": "Specify package versions",
        "pascal": "UnpinnedDependencies",
        "description": """A `run:` step exhibits this smell only if ALL conditions are true: (1) it installs external packages, tools, or dependencies using commands such as `apt install`, `apt-get install`, `brew install`, `pip install`, `pip install -r`, `npm install -g`, `gem install`, `cargo install`, or similar; (2) at least one installed package/tool is not explicitly pinned to a fixed version, commit SHA, digest, checksum-verified artifact, immutable release URL, or fixed version constraint; and (3) the installation can affect build, test, analysis, deployment, or release behavior. Treat dependency files such as `requirements.txt` as unpinned unless their contents are available and every dependency is pinned to an exact version. Do not flag package index updates without installation, installs from pinned commits/digests/immutable releases, installs with exact version constraints, or cases governed by verified lockfiles such as `package-lock.json`, `yarn.lock`, `poetry.lock`, `Pipfile.lock`, `Gemfile.lock`, or equivalent. When a referenced dependency file cannot be inspected, flag it as unpinned.""",
    },
    {
        "name": "Run CI on multiple language versions",
        "pascal": "SingleLangVersionCI",
        "description": """A CI job exhibits this smell only if ALL conditions are true: (1) the job performs CI validation, such as testing, building, linting, static analysis, or integration checks; (2) the job configures exactly one explicit project language or runtime version in the workflow file; and (3) the job does not define a matrix or equivalent strategy that runs the same CI validation on two or more distinct project language/runtime versions. This smell concerns only project language or runtime versions, not operating systems. Do not treat `runs-on`, OS labels such as `ubuntu-latest`, `windows-latest`, or `macos-latest`, or OS matrices as runtime-version coverage. Treat versions configured through project runtime setup actions, such as `actions/setup-node` with `node-version`, `actions/setup-python` with `python-version`, `actions/setup-java` with `java-version`, `actions/setup-go` with `go-version`, `ruby/setup-ruby` with `ruby-version`, or equivalent setup actions, as explicit runtime versions. If a matrix exists but contains only one distinct runtime version, still flag the job. Do not flag non-CI jobs, deployment-only jobs, documentation jobs, jobs already testing two or more distinct runtime versions, jobs without an explicit project runtime version in the workflow file, jobs that configure only auxiliary tool versions, or jobs whose CI purpose or runtime-version configuration cannot be determined confidently. When uncertain, do not flag. Flag one occurrence for each CI job that satisfies all conditions.""",
    },
]

# ─── Prompt Fragments ─────────────────────────────────────────────────────────

TARGET_SMELL_BLOCK = """<target_smell name="{smell_name}">
{smell_description}
</target_smell>"""

REASONING_BLOCK = (
    """<reasoning>Let's think step by step before emitting the structured output.</reasoning>"""
)

# ─── Inputs blocks ────────────────────────────────────────────────────────────
# Each system template declares the XML tags it will receive in the user turn.
# These are literal tag names (not {format} slots), so runtime data never enters
# the system prompt — only its contract is documented here.

# `<repository_context>` carries the workflow's repository metadata
# (current_repository, repository_owner, default_branch, workflow_file_path); it
# accompanies the workflow YAML in every role and is omitted when the ground_truth entry
# lacks it. Some smell definitions reference these values (e.g. comparing an
# action's owner/repo against current_repository to decide if it is external).
_REPO_CONTEXT_INPUT = (
    "- `<repository_context>`: the workflow's repository metadata — "
    "current_repository (owner/repo), repository_owner, default_branch, and "
    "workflow_file_path. Absent if the entry carries no repository metadata."
)

D1_INPUTS_BLOCK = f"""<inputs>
The user message contains, wrapped in XML tags:
- `<workflow>`: the GitHub Actions workflow YAML to analyze.
{_REPO_CONTEXT_INPUT}
</inputs>"""

D2_DETECTION_INPUTS_BLOCK = f"""<inputs>
The user message contains, wrapped in XML tags:
- `<workflow>`: the GitHub Actions workflow YAML to analyze.
{_REPO_CONTEXT_INPUT}
</inputs>"""

DETECTION_VALIDATION_INPUTS_BLOCK = f"""<inputs>
The user message contains, wrapped in XML tags:
- `<workflow>`: the GitHub Actions workflow YAML to analyze.
{_REPO_CONTEXT_INPUT}
- `<peer_detection_result>`: the Detection Agent's structured result to validate.
</inputs>"""

D2_REFACTOR_INPUTS_BLOCK = f"""<inputs>
The user message contains, wrapped in XML tags:
- `<original_workflow>`: the GitHub Actions workflow YAML to refactor.
{_REPO_CONTEXT_INPUT}
- `<detection_result>`: the Detection Agent's structured result.
</inputs>"""

REFACTOR_VALIDATION_INPUTS_BLOCK = f"""<inputs>
The user message contains, wrapped in XML tags:
- `<original_workflow>`: the original GitHub Actions workflow YAML.
{_REPO_CONTEXT_INPUT}
- `<detection_result>`: the converged detection result.
- `<proposed_refactor>`: the Refactor Agent's patched_workflow and justification to validate.
</inputs>"""

# Detection rules are intentionally decomposed into small fragments so D1, D2
# Detection, and the Detection Validation contract reuse exactly the same wording.
DETECTION_PRESENCE_RULE = """- smell_detected=true iff the workflow contains concrete evidence satisfying all required conditions of the target smell definition above. Similar-but-not-matching patterns count as smell_detected=false."""

DETECTION_JUSTIFICATION_RULE = """- detection_justification is a 1-4 sentence explanation of the detection decision, written in natural language and grounded in concrete evidence from the workflow. Reference specific keys, values, job names, step names, trigger names, or line numbers inline as needed, but include no YAML blocks, code snippets, or markdown fences inside the justification text."""

DETECTION_COUNT_RULE = """- count is the total number of instances of the target smell in the workflow. count=0 iff smell_detected=false; count>=1 iff smell_detected=true."""

# NO count-by-scope taxonomy (deliberate, 2026-05-29). `count` stays a plain
# instance count with no prescribed per-scope unit, left to the model's judgment.
# Rationale:
#   1. The scored metric is BINARY presence (precision/recall/f1 over
#      ground_truth_positive); smell_count is recorded but never enters the metric, so
#      count is diagnostic only.
#   2. The human baseline's counts are sparse/incomplete (17/863 present rows
#      exceed 1; 75 present rows carry no count) — too weak to anchor a rule.
#   3. Deriving a scope→smell counting convention from gha-ci-detector would
#      import the tool's identification bias into the baseline-vs-ours comparison,
#      so the agents' counting is NOT anchored to the tool.

DETECTION_EVIDENCE_RULES = """- evidence_ranges lists one {start_line, end_line} object, using 1-based inclusive line numbers, for every instance that can be localized to concrete workflow lines. List ranges in order of appearance.

- For absence-based smells where the missing construct has no direct line range, use the range of the closest affected enclosing element, such as the job, step, trigger, or workflow-level block. If no meaningful enclosing range can be identified, use evidence_ranges=[].

- evidence_ranges is supporting evidence, not a per-instance count: it may hold more or fewer ranges than count, and it MAY be non-empty even when count=0 (point to the enclosing block you inspected to conclude the smell is absent). count alone carries the instance total."""

DETECTION_CONFIDENCE_RULES = """- detection_confidence MUST be one of: low, medium, high.
  - Use high when the workflow clearly satisfies or clearly does not satisfy the target smell definition.
  - Use medium when the decision is supported but depends on contextual interpretation, such as whether a job is truly a deployment, test, CI, privileged job, release job, or publish job.
  - Use low when the workflow is incomplete, heavily templated, ambiguous, or lacks enough context to confidently assess the smell.

- detection_confidence_justification is a 1-4 sentence explanation of why the confidence level was assigned."""

DETECTION_CORE_RULES = "\n\n".join(
    [
        DETECTION_PRESENCE_RULE,
        DETECTION_JUSTIFICATION_RULE,
        DETECTION_COUNT_RULE,
        DETECTION_EVIDENCE_RULES,
        DETECTION_CONFIDENCE_RULES,
    ]
)

# Detection validation reuses the same detection core rules but keeps its
# arbitration rules separate.
DETECTION_VALIDATION_ARBITRATION_RULES = """- First form your own judgment by analyzing the workflow against all required conditions of the target smell definition. Do not rely on the peer's result before completing your independent assessment.

{detection_core_rules}

- agrees=true iff all of the following conditions hold:
  1. your independent smell_detected matches the peer's smell_detected;
  2. your independent count matches the peer's count;
  3. the peer's evidence_ranges are materially consistent with yours, referring to the same affected workflow constructs even if line boundaries differ slightly because of enclosing-block interpretation;
  4. the peer's detection_justification is supported by concrete workflow evidence.

- agrees=false if you disagree with the peer's smell_detected value, count, evidence_ranges, or if the peer's justification is not supported by the workflow evidence.

- If agrees=true, validated_result must mirror the peer_detection_result verbatim and disagreement_feedback must be null.

- If agrees=false, validated_result must contain your corrected structured detection result with reasoning, target_smell, smell_detected, detection_justification, count, evidence_ranges, detection_confidence, and detection_confidence_justification.

- If agrees=false, disagreement_feedback must be a concrete 1-4 sentence explanation that the Detection Agent can use in the next iteration. Explain what was missed, over-counted, under-counted, incorrectly localized, or inferred without sufficient evidence.

- detection_justification, detection_confidence_justification, and disagreement_feedback must be written in natural language and grounded in concrete workflow evidence. Reference specific keys, values, job names, step names, trigger names, or line numbers inline as needed, but include no YAML blocks, code snippets, or markdown fences.

- validation_confidence MUST be one of: low, medium, high.
  - Use high when your agreement or disagreement with the peer result is clear from the workflow.
  - Use medium when the decision depends on contextual interpretation, such as whether a job is truly a deployment, test, CI, privileged job, release job, or publish job.
  - Use low when the workflow is incomplete, heavily templated, ambiguous, or the peer result cannot be reliably assessed from the available context.

- validation_confidence_justification is a 1-4 sentence explanation of why the validation confidence level was assigned.

- Do not suggest refactoring changes. This agent validates detection only.

- When you emit a fresh justification or disagreement_feedback, write it as a 1-4 sentence explanation in natural language grounded in concrete evidence from the workflow. Reference specific keys, values, or line numbers inline as needed, but include no YAML blocks, code snippets, or markdown fences inside the text."""

D1_REFACTOR_RULES = """- When smell_detected=false, patched_workflow is the original workflow verbatim.

- When smell_detected=true, patched_workflow is the full workflow with the smallest intent-preserving edit that removes ALL instances of the target smell. The edit should preserve the workflow's intended purpose while changing only the behavior necessary to remove the target smell.

- patched_workflow MUST be syntactically valid YAML and should avoid constructs known to be invalid in GitHub Actions workflow files.

- Do not introduce unrelated changes, stylistic rewrites, formatting-only edits, or fixes for other smells.

- patched_workflow contains ONLY the raw YAML content. Never wrap it in `<workflow>` tags, markdown code fences, or any other delimiters. Emit the YAML body directly.

- refactor_justification is a 1-4 sentence explanation written in natural language and grounded in concrete evidence from the edit you made: which keys, jobs, steps, or triggers you changed and why the change removes the target smell while preserving the workflow's intended purpose. Reference specific keys, values, job names, step names, or line numbers inline as needed, but include no YAML blocks, code snippets, or markdown fences inside the text.

- When smell_detected=false, refactor_justification is an empty string.

- refactor_confidence MUST be one of: low, medium, high when smell_detected=true, and null when smell_detected=false.
  - Use high when the edit is localized, clearly removes all detected instances of the target smell, and is unlikely to change the workflow's intended behavior.
  - Use medium when the edit likely removes the smell but depends on contextual assumptions, such as required permissions, deployment conditions, cache semantics, action-version compatibility, runner availability, or branch/event semantics.
  - Use low when the workflow is incomplete, heavily templated, ambiguous, or the smell can only be removed through a change that may alter workflow behavior.

- refactor_confidence_justification is a 1-4 sentence explanation of why the confidence level was assigned, focusing on completeness, safety, and behavior preservation. When smell_detected=false, refactor_confidence_justification is an empty string."""

REFACTOR_CORE_RULES = """- Use the detection_result to prioritize affected locations. When evidence_ranges are provided, anchor your edits on those locations. When evidence_ranges=[] because the smell is absence-based, add the missing construct at the appropriate workflow-level, trigger-level, job-level, or step-level scope according to the smell definition.

- Do not blindly trust the detection_result if the workflow context shows that an additional location must be changed to remove all instances of the target smell. However, do not fix unrelated smells.

- Never introduce unrelated changes, stylistic rewrites, formatting-only edits, reordering, renaming, or fixes for other smells.

- patched_workflow is the full workflow after applying the smallest intent-preserving edit that removes EVERY detected instance of the target smell.

- patched_workflow MUST be syntactically valid YAML and should conform to the GitHub Actions workflow schema.

- patched_workflow contains ONLY the raw YAML content. Never wrap it in tags, markdown code fences, or any other delimiters. Emit the YAML body directly.

- refactor_justification is a 1-4 sentence explanation written in natural language and grounded in the concrete edit made. Explain which keys, jobs, steps, triggers, references, or line ranges were changed and why the edit removes the target smell while preserving the workflow's intended purpose. Include no YAML blocks, code snippets, or markdown fences inside the justification text.

- refactor_confidence MUST be one of: low, medium, high.
  - Use high when the edit is localized, clearly removes all detected instances of the target smell, and is unlikely to change the workflow's intended behavior.
  - Use medium when the edit likely removes the smell but depends on contextual assumptions, such as required permissions, deployment conditions, cache semantics, action-version compatibility, runner availability, branch filters, or event semantics.
  - Use low when the workflow is incomplete, heavily templated, ambiguous, or the smell can only be removed through a change that may alter workflow behavior.

- refactor_confidence_justification is a 1-4 sentence explanation of why the confidence level was assigned, focusing on completeness, safety, and behavior preservation."""

D2_REFACTOR_GUARD_RULES = """- This Refactoring Agent is invoked only when the Detection Agent predicted smell_detected=true; assume the target smell is present and produce a refactoring that removes it."""

REFACTOR_VALIDATION_RULES = """- First inspect the proposed patched_workflow against the target smell definition. Do not rely only on the Refactor Agent's justification.

- agrees=true iff all of the following conditions hold:
  1. the patched_workflow no longer contains any instance of the target smell;
  2. remaining_count=0;
  3. the patched_workflow is syntactically valid YAML;
  4. the patched_workflow appears to conform to the GitHub Actions workflow schema;
  5. the patch does not introduce unrelated changes, formatting-only rewrites, or fixes for other smells;
  6. the patch preserves the original workflow's intended behavior as much as can be assessed from the workflow context.

- If agrees=true, disagreement_feedback must be null.

- agrees=false if any target-smell instance remains, the YAML is invalid, the workflow schema appears invalid, the patch changes unrelated workflow behavior, or the patch does not address the detection_result.

- remaining_count is the number of target-smell instances still present in the patched_workflow. Set remaining_count=0 iff no target-smell instance remains.

- If agrees=false, disagreement_feedback must be a concrete 1-4 sentence explanation that the Refactor Agent can use in the next iteration. Explain whether the problem is an unresolved smell, invalid YAML, invalid GitHub Actions structure, behavior-changing edit, unrelated change, or incomplete use of the detection_result.

- When disagreement_feedback refers to unresolved target-smell instances, ground it in concrete workflow evidence. Reference specific keys, values, job names, step names, trigger names, or line numbers inline as needed, but include no YAML blocks, code snippets, or markdown fences.

- When you populate disagreement_feedback, write it as a 1-4 sentence explanation in natural language grounded in concrete evidence from the refactored workflow. Reference specific keys, values, or line numbers inline as needed, but include no YAML blocks, code snippets, or markdown fences inside the text.

- validation_confidence MUST be one of: low, medium, high.
  - Use high when acceptance or rejection is clear from the patched workflow.
  - Use medium when the decision depends on contextual assumptions, such as permissions, action compatibility, deployment intent, cache behavior, event semantics, or branch/path filtering.
  - Use low when the workflow is incomplete, heavily templated, ambiguous, or the patch cannot be reliably assessed from the available context.

- validation_confidence_justification is a 1-4 sentence explanation of why the validation confidence level was assigned.

- Do not propose a full replacement patch. Provide feedback only."""

# Each <output_format> block is rendered from its OutputContract (output_contract.py)
# — the single declarative source of the field order + prose class-name. The
# rendered bytes are byte-identical to the previously frozen literals (guarded by
# the PROMPT_HASHES fingerprint test); agents.py round-trips each contract against
# its bound Pydantic schema at import.
COMBINED_OUTPUT_FORMAT = OUTPUT_CONTRACTS["combined"].render()

DETECTION_OUTPUT_FORMAT = OUTPUT_CONTRACTS["detection"].render()

DETECTION_VALIDATION_OUTPUT_FORMAT = OUTPUT_CONTRACTS["detection_validation"].render()

REFACTOR_OUTPUT_FORMAT = OUTPUT_CONTRACTS["refactor"].render()

REFACTOR_VALIDATION_OUTPUT_FORMAT = OUTPUT_CONTRACTS["refactor_validation"].render()

# ─── System Prompt Templates ─────────────────────────────────────────────────

D1_COMBINED_TEMPLATE = """<role>You are a senior GitHub Actions engineer with many years of CI/CD experience, responsible for identifying whether a target configuration smell is present or absent in a workflow and, only when present, refactoring the workflow to remove it.</role>

<task>Your task is to inspect a GitHub Actions workflow for the target configuration smell below. First, decide whether the smell is present. If the smell is present, produce a refactored version that removes all instances of the target smell. If the smell is absent, return the original workflow verbatim.</task>

{target_smell_block}

{d1_inputs_block}

<decision_rules>
{detection_core_rules}

{d1_refactor_rules}
</decision_rules>

{reasoning_block}

{combined_output_format}"""

D2_DETECTION_TEMPLATE = """<role>You are a senior GitHub Actions engineer with many years of CI/CD experience, responsible for identifying whether a target smell is present or absent in a workflow.</role>

<task>Your task is to inspect a GitHub Actions workflow and decide, in a single shot, whether the target configuration smell below is present or absent. There is no peer reviewer, no validation agent, no feedback loop, and no refactoring in this step. Your verdict is final.</task>

{target_smell_block}

{d2_detection_inputs_block}

<decision_rules>
{detection_core_rules}
</decision_rules>

{reasoning_block}

{detection_output_format}"""

D2_REFACTOR_TEMPLATE = """<role>You are a senior GitHub Actions engineer with many years of CI/CD experience, responsible for refactoring workflows to remove a specific configuration smell while preserving the workflow's intended behavior.</role>

<task>Your task is to refactor the original GitHub Actions workflow only when the Detection Agent has predicted that the target smell is present. Use the detection result as guidance, but perform the smallest intent-preserving edit that removes all detected instances of the target smell. There is no validation agent, no peer reviewer, and no feedback loop.</task>

{target_smell_block}

{d2_refactor_inputs_block}

<decision_rules>
{d2_refactor_guard_rules}

{refactor_core_rules}
</decision_rules>

{reasoning_block}

{refactor_output_format}"""

# ─── Builder API ──────────────────────────────────────────────────────────────

# User-turn building moved to :mod:`gha_agents.user_turn` (the single source
# for the data-demarcation contract). The denylist below — the names that must
# never appear as ``{slot}`` placeholders in a *system* template, because they
# carry runtime data — is derived from that one source: every user-turn tag plus
# every output-contract field name. No hand-maintained list to drift.
DYNAMIC_USER_PLACEHOLDERS = all_user_turn_tags() | {
    field for contract in OUTPUT_CONTRACTS.values() for field in contract.fields
}


def _extract_placeholders(template: str) -> set[str]:
    return {field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name}


def _validate_system_template(template: str, *, template_name: str) -> None:
    placeholders = _extract_placeholders(template)
    invalid = placeholders & DYNAMIC_USER_PLACEHOLDERS
    if invalid:
        raise ValueError(
            f"{template_name} contains dynamic user placeholders that must be moved to a user prompt: {sorted(invalid)}"
        )


def _get_smell_definition(target_smell_name: str) -> tuple[str, str]:
    """Return ``(name, description)`` from SMELL_CATALOG for template interpolation."""
    entry = next((s for s in SMELL_CATALOG if s["name"] == target_smell_name), None)
    if entry is None:
        raise ValueError(f"Unknown target smell: {target_smell_name!r}")
    return entry["name"], entry["description"]


def _target_smell_block(target_smell_name: str) -> str:
    name, description = _get_smell_definition(target_smell_name)
    return TARGET_SMELL_BLOCK.format(smell_name=name, smell_description=description)


def _render_system_prompt(template: str, *, target_smell_name: str, template_name: str) -> str:
    _validate_system_template(template, template_name=template_name)
    detection_validation_rules = DETECTION_VALIDATION_ARBITRATION_RULES.format(
        detection_core_rules=DETECTION_CORE_RULES
    )
    return template.format(
        target_smell_block=_target_smell_block(target_smell_name),
        d1_inputs_block=D1_INPUTS_BLOCK,
        d2_detection_inputs_block=D2_DETECTION_INPUTS_BLOCK,
        detection_validation_inputs_block=DETECTION_VALIDATION_INPUTS_BLOCK,
        d2_refactor_inputs_block=D2_REFACTOR_INPUTS_BLOCK,
        refactor_validation_inputs_block=REFACTOR_VALIDATION_INPUTS_BLOCK,
        detection_core_rules=DETECTION_CORE_RULES,
        d1_refactor_rules=D1_REFACTOR_RULES,
        detection_validation_rules=detection_validation_rules,
        d2_refactor_guard_rules=D2_REFACTOR_GUARD_RULES,
        refactor_core_rules=REFACTOR_CORE_RULES,
        refactor_validation_rules=REFACTOR_VALIDATION_RULES,
        reasoning_block=REASONING_BLOCK,
        combined_output_format=COMBINED_OUTPUT_FORMAT,
        detection_output_format=DETECTION_OUTPUT_FORMAT,
        detection_validation_output_format=DETECTION_VALIDATION_OUTPUT_FORMAT,
        refactor_output_format=REFACTOR_OUTPUT_FORMAT,
        refactor_validation_output_format=REFACTOR_VALIDATION_OUTPUT_FORMAT,
    )


def build_combined_prompt(target_smell_name: str) -> str:
    return _render_system_prompt(
        D1_COMBINED_TEMPLATE,
        target_smell_name=target_smell_name,
        template_name="D1_COMBINED_TEMPLATE",
    )


_DETECTION_TEMPLATES: dict[Scenario, str] = {
    "D2": D2_DETECTION_TEMPLATE,
}

_REFACTOR_TEMPLATES: dict[Scenario, str] = {
    "D2": D2_REFACTOR_TEMPLATE,
}


def build_detection_prompt(target_smell_name: str, scenario: Scenario) -> str:
    """Render the detection system prompt for ``scenario``."""
    if scenario not in _DETECTION_TEMPLATES:
        raise ValueError(
            f"Detection prompt is only defined for D2, got scenario={scenario!r}."
        )
    return _render_system_prompt(
        _DETECTION_TEMPLATES[scenario],
        target_smell_name=target_smell_name,
        template_name=f"{scenario}_DETECTION_TEMPLATE",
    )


def build_refactor_prompt(target_smell_name: str, scenario: Scenario) -> str:
    """Render the refactor system prompt for ``scenario``."""
    if scenario not in _REFACTOR_TEMPLATES:
        raise ValueError(
            f"Refactor prompt is only defined for D2, got scenario={scenario!r}."
        )
    return _render_system_prompt(
        _REFACTOR_TEMPLATES[scenario],
        target_smell_name=target_smell_name,
        template_name=f"{scenario}_REFACTOR_TEMPLATE",
    )


# ─── Hashing ──────────────────────────────────────────────────────────────────


def prompt_hash(template: str) -> str:
    """Return the full SHA-256 hex digest of a rendered prompt."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


_SCENARIO_ROLE_BUILDERS: dict[Scenario, dict[str, Callable[[str], str]]] = {
    "D1": {"combined": build_combined_prompt},
    "D2": {
        "detection": lambda s: build_detection_prompt(s, "D2"),
        "refactor": lambda s: build_refactor_prompt(s, "D2"),
    },
}


def get_all_prompt_hashes(scenario: Scenario) -> dict[str, dict[str, str]]:
    """Return ``{role: {smell_pascal: sha256}}`` for every (role, smell) pair in the scenario."""
    if scenario not in _SCENARIO_ROLE_BUILDERS:
        raise ValueError(
            f"Unknown scenario: {scenario!r}. Expected one of {list(_SCENARIO_ROLE_BUILDERS)}."
        )
    return {
        role: {entry["pascal"]: prompt_hash(builder(entry["name"])) for entry in SMELL_CATALOG}
        for role, builder in _SCENARIO_ROLE_BUILDERS[scenario].items()
    }


PROMPT_HASHES: dict[Scenario, dict[str, dict[str, str]]] = {
    scenario: get_all_prompt_hashes(scenario) for scenario in ("D1", "D2")
}


# ─── Manual prompt analysis dump ──────────────────────────────────────────────
# Run `python -m gha_agents.prompts [ground_truth.json]` to write one file per
# (scenario, role) at docs/prompts/{scenario}_{role}.md: the two messages the
# model receives for a representative smell — the system prompt wrapped in
# <system_prompt> and the typed user turn wrapped in <user_prompt>. Only the
# <target_smell> block (inside <system_prompt>) varies across the 22 catalog
# smells, so one is enough.
#
# The <user_prompt> is grounded in a REAL ground_truth entry (not a mock): its
# <workflow> and <repository_context> come verbatim from an entry of the given
# ground_truth (default: the package's own ground_truth-smoke.json), chosen so the entry
# actually exhibits the representative smell. The <detection_result> /
# <proposed_refactor> blocks are agent OUTPUTS (absent from any ground_truth), so they
# are synthesized from the real Pydantic models — but grounded in the chosen
# entry: real target_smell, real smell-present verdict, and real instance count
# from its present_smells. agents + user_turn are imported lazily so module load
# stays free of a prompts<->agents cycle.

# The smoke ground_truth is the default doc source (tracked, sampled from final_ground_truth).
# Override with a CLI arg, e.g. `python -m gha_agents.prompts ground_truth-smoke.json`.
_DEFAULT_DOC_GROUND_TRUTH = Path(__file__).parent.parent / "ground_truth-smoke.json"


def _smell_instance_count(entry: dict, pascal: str) -> int | None:
    """Instance count for ``pascal`` if the entry lists it present, else None.

    Reads ``present_smells`` (each ``{pascal, count}``) so the detection example
    mirrors the ground_truth's ground truth for that smell. A smell that is absent or
    unlisted returns None (the example then renders it as not-detected).
    """
    for smell in entry.get("present_smells") or []:
        if isinstance(smell, dict) and smell.get("pascal") == pascal:
            count = smell.get("count")
            return count if isinstance(count, int) and count > 0 else 1
    return None


def _select_doc_entry(entries: list[dict], pascal: str) -> dict:
    """Pick a real entry that exhibits ``pascal`` (grounded detection example),
    falling back to the first entry when none do."""
    for entry in entries:
        if _smell_instance_count(entry, pascal) is not None:
            return entry
    return entries[0]


def _example_user_turns(entry: dict, rep_smell: str) -> dict[tuple[str, str], str]:
    """Return the user turn for each (scenario, role), grounded in ``entry``.

    ``<workflow>`` and ``<repository_context>`` are the entry's real values. The
    structured ``<detection_result>`` / ``<proposed_refactor>`` payloads are
    serialized from real model instances exactly as :mod:`gha_agents.teams`
    does (so the shapes cannot drift from the live schema), with the verdict and
    count taken from the entry's ground-truth labels.
    """
    from .agents import DetectionContract, ValidSmells
    from .user_turn import (
        D2_REFACTOR_BUILDER,
        WORKFLOW_BUILDER,
        repository_context_from_entry,
    )

    smell = ValidSmells(rep_smell)
    wf = entry["yaml_content"]
    repo_ctx = repository_context_from_entry(entry)
    count = _smell_instance_count(entry, smell.name)
    detected = count is not None
    repo = entry.get("repository") or "the repository"

    detection = DetectionContract(
        reasoning="(the agent's private chain of thought)",
        target_smell=smell,
        smell_detected=detected,
        detection_justification=(
            f"At least one job in {repo} uses an external action via a step-level "
            "uses: and declares no permissions block at the workflow or job level."
            if detected
            else f"No job in {repo} exhibits the target smell after checking each "
            "against its required conditions."
        ),
        count=count or 0,
        # MissingPermExternalActions is absence-based, so no direct line range.
        evidence_ranges=[],
        detection_confidence="high",
        detection_confidence_justification=(
            "The presence or absence of a permissions block is unambiguous in the workflow."
        ),
    )
    detection_json = detection.model_dump_json(indent=2)

    return {
        ("D1", "combined"): WORKFLOW_BUILDER.build(workflow=wf, repository_context=repo_ctx),
        ("D2", "detection"): WORKFLOW_BUILDER.build(workflow=wf, repository_context=repo_ctx),
        ("D2", "refactor"): D2_REFACTOR_BUILDER.build(
            original_workflow=wf, repository_context=repo_ctx, detection_result=detection_json
        ),
    }


def _dump_prompts_for_analysis(ground_truth_path: str | Path | None = None) -> None:
    """Render the system/user prompt pair for every (scenario, role) to docs/prompts.

    Grounds each ``<user_prompt>`` in a real ground_truth entry so the dumped docs show
    the actual data demarcation. Runs only under ``python -m gha_agents.prompts``.
    """
    rep_smell = SMELL_CATALOG[0]["name"]
    from .agents import ValidSmells

    ground_truth_path = Path(ground_truth_path) if ground_truth_path else _DEFAULT_DOC_GROUND_TRUTH
    entries = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    entry = _select_doc_entry(entries, ValidSmells(rep_smell).name)
    user_turns = _example_user_turns(entry, rep_smell)

    output_dir = Path(__file__).parent.parent / "docs" / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"ground_truth={ground_truth_path} entry={entry.get('file_key')!r} smell={ValidSmells(rep_smell).name}"
    )
    for scenario in ("D1", "D2"):
        for role, builder in _SCENARIO_ROLE_BUILDERS[scenario].items():
            document = (
                f"<system_prompt>\n{builder(rep_smell)}\n</system_prompt>\n\n"
                f"<user_prompt>\n{user_turns[(scenario, role)]}\n</user_prompt>\n"
            )
            path = output_dir / f"{scenario}_{role}.md"
            path.write_text(document, encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    import sys

    _dump_prompts_for_analysis(sys.argv[1] if len(sys.argv) > 1 else None)
