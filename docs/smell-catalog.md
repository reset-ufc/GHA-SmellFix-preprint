# Operationalized smell catalog

The complete, refined catalog of the **19 GitHub Actions configuration smells** used in
this study. The paper's smell table (`tab:smell-catalog`) reports only the *core operational
criterion* per smell; this file is the full definition it points to.

Each smell is operationalized along the four axes the study evaluates:

- **Detection criterion**: the authoritative `present` / `absent` rule used to label a
  `(workflow, smell)` pair. This is the same text the detection agents and the human
  annotators were given.
- **Refactoring expectation**: what a correct fix is expected to do.
- **Expected constructs**: the GitHub Actions constructs a fix may legitimately add,
  change, or inspect (from Bardsiri et al.'s construct abstraction, restricted to the 117
  constructs relevant to these 19 smells). These bound the *scope* of an acceptable fix;
  they are **not** a fixed refactoring template; a smell can be removed by more than one
  valid edit.
- **Safety**: the common ways a fix can remove the smell yet damage the workflow. Flagged
  so refactor outputs are judged construct-aware, not just on smell removal.

The catalog is derived from Khatami et al.'s catalog (22 smells; the three readability /
pure-formatting smells (*AddCommentsToWorkflows*, *CorrectIndentation*, *NameRunSteps*) were excluded). The numbering `1-19` here is shared verbatim with
[`validation/detection-guide.md`](validation/detection-guide.md) (worked positive/negative
examples per smell). The refactoring codebook,
[`validation/refactoring-codebook.md`](validation/refactoring-codebook.md), applies the same definitions.

> **Source of truth.** The prose detection criteria below are the labeling authority. Where
> a construct path uses `<job_id>` it means "any job id"; `steps[*]` means "any step".

---

### 1. MissingPermExternalActions: *Define permissions for workflows with external actions*

**Detection criterion.** A job exhibits this smell only if ALL conditions are true: (1) it contains at least one step-level `uses:` on a `job`; (2) at least one reference is a GitHub action in the form `owner/repo@ref` or `owner/repo/path@ref`; (3) the `owner/repo` portion differs from the provided `current_repository`; and (4) the job has no effective `permissions:` configuration, meaning there is neither a workflow-level nor job-level `permissions:` block. Treat relative-path references, such as `uses: ./.github/actions/setup`, as local actions. Do not flag local actions, same-repository action references, job-level `uses:` references to reusable workflows, Docker image references such as `uses: docker://alpine:3.19`, unparsable references, or jobs with any applicable `permissions:` block, even if permissions are broad. Flag one occurrence per job, regardless of how many external step-level actions it contains.

**Refactoring expectation.** Add a `permissions:` block (workflow- or job-level) that grants only the scopes the job actually needs (least privilege).

**Expected constructs (30).** `permissions.<scope>` and `jobs.<job_id>.permissions.<scope>` for `<scope>` in {`contents`, `id-token`, `pull-requests`, `issues`, `packages`, `actions`, `checks`, `statuses`, `pages`, `security-events`, `deployments`, `attestations`, `discussions`, `models`}, plus the bare `permissions` / `jobs.<job_id>.permissions` roots. Example: `jobs.<job_id>.permissions.contents`.

**Safety.** A fix that adds `permissions: write-all` or broad scopes removes the smell but *adds* a security risk; least privilege is the point. Grant the minimum.

### 2. UnpinnedActionVersion: *Use commit hash instead of tags for action versions*

**Detection criterion.** A step-level `uses:` reference exhibits this smell only if ALL conditions are true: (1) the reference appears under `jobs.<job_id>.steps[*].uses`; (2) the `uses:` value is a GitHub action reference in the form `owner/repo@ref` or `owner/repo/path@ref`; and (3) `ref` is not exactly a 40-character hexadecimal commit SHA. Flag branch names, tags, major-version aliases, and floating references such as `@main`, `@master`, `@develop`, `@v1`, `@v2`, `@latest`, or any other non-40-character hexadecimal reference. This applies to external actions and to same-repository actions when they are referenced through the explicit `owner/repo@ref` or `owner/repo/path@ref` form. Do not flag local actions referenced by relative path, such as `uses: ./.github/actions/setup`, because they do not use an explicit `@ref`. Do not flag actions pinned to a full 40-character hexadecimal commit SHA, `uses:` values without `@ref`, non-GitHub Docker image references such as `uses: docker://alpine:3.19`, job-level `uses:` references to reusable workflows, including local workflows such as `jobs.<job_id>.uses: ./.github/workflows/ci.yml`, or references whose value cannot be parsed confidently. Flag one occurrence for each step-level `uses:` reference that satisfies all conditions.

**Refactoring expectation.** Repin the action to a full 40-character commit SHA (`uses: owner/repo@<sha>`).

**Expected constructs (1).** `jobs.<job_id>.steps[*].uses`.

**Safety.** The SHA must be a real, resolvable commit for that action; a fabricated ("hallucinated") SHA silently breaks the run. Do not switch the action or bump its major version while pinning.

### 3. MissingTokenPermissions: *Set permissions for GitHub Token*

**Detection criterion.** A job exhibits this smell only if ALL conditions are true: (1) the job uses the GitHub-provided token explicitly through `secrets.GITHUB_TOKEN` or `github.token`, or uses an action/command that clearly performs authenticated GitHub operations such as creating comments, labels, releases, deployments, packages, statuses, pull request updates, issue updates, or repository writes; (2) the job has no effective `permissions:` configuration, either from a workflow-level `permissions:` block or a job-level `permissions:` block. Do not flag jobs with an applicable workflow-level or job-level `permissions:` block, even if the permissions are broad. Do not flag jobs only because they run on GitHub Actions, use `actions/checkout`, or perform ordinary build/test steps. This smell concerns only the absence of explicit token permission configuration, not whether the configured permissions follow least privilege. When token use is uncertain, do not flag.

**Refactoring expectation.** Add a `permissions:` block granting only the scopes the token actually uses.

**Expected constructs (30).** Same permission-path family as smell 1. Example: `permissions.issues`.

**Safety.** As with smell 1, never resolve this by granting `write-all`. Scope to what the authenticated operations require.

---

### 4. MissingForkGuardPR: *Prevent running issue/PR actions on forks*

**Detection criterion.** A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by an issue- or pull-request-related event that may involve forked repositories, such as `pull_request`, `pull_request_target`, `issue_comment`, `pull_request_review`, or `pull_request_review_comment`; (2) at least one job or relevant step performs an issue or pull request action, such as commenting, labeling, approving, updating statuses, or modifying issue/PR metadata; and (3) that job or step has no effective fork guard. Effective fork guards must clearly prevent the issue/PR action from running for forked pull requests, such as `github.event.pull_request.head.repo.fork == false`, `github.event.pull_request.head.repo.full_name == github.repository`, or equivalent same-repository PR checks. For `issue_comment`, the guard must also ensure the comment belongs to a pull request before checking the associated PR fork status. Do not treat repository-identity checks alone, label-based checks, branch checks, actor checks, event-type checks, or comment-content checks as fork guards unless combined with an explicit fork-status or same-repository PR check. Do not flag workflows that are not triggered by issue/PR-related events, do not perform issue/PR actions, are protected by effective fork guards, or cannot be determined confidently. Flag one occurrence for each workflow that satisfies all conditions.

**Refactoring expectation.** Add an effective fork guard (`if:`) so the issue/PR action does not run for forked pull requests.

**Expected constructs (2).** `jobs.<job_id>.if`, `jobs.<job_id>.steps[*].if`.

**Safety.** The guard must actually block forks: a fork-status or same-repository check. An owner-only, label, or actor check is *not* a real guard and leaves the smell. Do not remove the action's intended function.

### 5. MissingJobTimeout: *Avoid jobs without timeouts*

**Detection criterion.** A job exhibits this smell only if ALL conditions are true: (1) it is a job entry defined in the current workflow file; and (2) it does not define `timeout-minutes` at the job level. Flag one occurrence for each job that satisfies these conditions, even if one or more steps define their own `timeout-minutes`, and even if the job has no `steps` block. Step-level `timeout-minutes` does not count as a job-level timeout and does not prevent this smell. Do not flag jobs that explicitly define `timeout-minutes` at the job level or jobs whose structure or timeout configuration cannot be determined confidently. This rule checks only whether `timeout-minutes` is present at the job level in the current workflow file; it does not inspect called reusable workflows or their internal timeout configuration.

**Refactoring expectation.** Add `timeout-minutes` at the job level with a sensible bound.

**Expected constructs (1).** `jobs.<job_id>.timeout-minutes`.

**Safety.** A job that only *calls a reusable workflow* (`jobs.<job_id>.uses:`, no `steps`) does not accept `timeout-minutes` the same way; adding it there produces an invalid workflow that fails to run. This is a frequent risky-removal pattern in the annotated outputs; a safe fix must keep the workflow valid.

### 6. MissingConcurrencyPR: *Stop running workflows when there is a newer commit in PR*

**Detection criterion.** A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `pull_request` or `pull_request_target`; (2) it does not define an effective workflow-level `concurrency:` configuration with `cancel-in-progress: true`; and (3) at least one job that can run on the PR trigger does not define an effective job-level `concurrency:` configuration with `cancel-in-progress: true`. Flag one occurrence for each workflow where newer commits to a pull request may start additional workflow runs without canceling older in-progress runs for that PR. Do not flag workflows without `pull_request` or `pull_request_target` triggers. Do not flag workflows with workflow-level `concurrency.cancel-in-progress: true`, or workflows where every job that can run on the PR trigger has job-level `concurrency.cancel-in-progress: true`. For this smell, only check whether an applicable `concurrency:` block with `cancel-in-progress: true` exists. Do not evaluate whether the `concurrency.group` expression is optimal, pull-request-specific, branch-specific, ref-specific, workflow-specific, or otherwise well designed. Do not flag configurations whose `concurrency:` or `cancel-in-progress` value cannot be confidently interpreted. When uncertain, do not flag.

**Refactoring expectation.** Add a `concurrency:` block with `cancel-in-progress: true`, keyed per PR/ref (workflow- or job-level).

**Expected constructs (6).** `concurrency`, `concurrency.group`, `concurrency.cancel-in-progress`, and the `jobs.<job_id>.concurrency*` equivalents. Example: `concurrency.cancel-in-progress`.

**Safety.** Adding concurrency is low-risk. Do not choose a `group` so coarse that it serializes unrelated workflows or PRs.

### 7. MissingConcurrencyBranch: *Stop running workflows when there is a newer commit in branch*

**Detection criterion.** A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `push` events; (2) it does not define an effective workflow-level `concurrency:` configuration with `cancel-in-progress: true`; and (3) at least one job that can run on branch `push` events does not define an effective job-level `concurrency:` configuration with `cancel-in-progress: true`. Flag one occurrence for each workflow where newer commits pushed to a branch may start additional workflow runs without canceling older in-progress runs for that branch. Do not flag workflows that are not triggered by `push`, workflows triggered only by tag pushes, workflows with workflow-level `concurrency.cancel-in-progress: true`, or workflows where every job that can run on branch `push` events has job-level `concurrency.cancel-in-progress: true`. For this smell, only check whether an applicable `concurrency:` block with `cancel-in-progress: true` exists. Do not evaluate whether the `concurrency.group` expression is optimal, branch-specific, workflow-specific, or otherwise well designed. Do not flag configurations whose `concurrency:` or `cancel-in-progress` value cannot be confidently interpreted. When uncertain, do not flag.

**Refactoring expectation.** Add a `concurrency:` block with `cancel-in-progress: true`, keyed on the branch ref.

**Expected constructs (6).** Same `concurrency*` family as smell 6. Example: `jobs.<job_id>.concurrency.cancel-in-progress`.

**Safety.** Key the group on the branch ref; a group shared across branches cancels legitimate parallel runs.

### 8. MissingPathFilters: *Avoid running CI actions when no source code has changed*

**Detection criterion.** A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `push` or `pull_request`; (2) its main purpose is CI validation, such as build, test, lint, static analysis, packaging checks, or dependency checks; and (3) at least one relevant `push` or `pull_request` trigger does not define `paths` or `paths-ignore`. Flag one occurrence for each workflow whose relevant CI trigger lacks `paths` or `paths-ignore`. Do not flag workflows where every relevant `push` or `pull_request` trigger defines `paths` or `paths-ignore`, regardless of whether the filter is complete, precise, or optimal. Do not flag workflows whose main purpose is documentation, release, deployment, notification, scheduled maintenance, formatting-only automation, whole-repository automation, or cases where the workflow purpose cannot be determined confidently. When the trigger, workflow purpose, or path-filter configuration cannot be determined confidently, do not flag.

**Refactoring expectation.** Add `paths` or `paths-ignore` filters to the relevant CI trigger.

**Expected constructs (4).** `on.push.paths[*]`, `on.push.paths-ignore[*]`, `on.pull_request.paths[*]`, `on.pull_request.paths-ignore[*]`. Example: `on.push.paths[*]`.

**Safety.** Over-broad `paths-ignore` can silently skip CI for changes that need it. The filter must not exclude the source paths the workflow validates.

### 9. MissingForkGuardSchedule: *Avoid executing scheduled workflows on forks*

**Detection criterion.** A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `schedule`; (2) at least one scheduled job can run without a condition restricting execution to the source repository or repository owner; and (3) there is no effective workflow-level or job-level `if:` condition using values such as `github.repository`, `github.repository_owner`, or `github.event.repository.fork` to restrict scheduled execution. Do not flag workflows that are not triggered by `schedule`, or scheduled workflows where every scheduled job is protected by a repository or owner check, such as `github.repository == 'owner/repo'`, `github.repository_owner == 'owner'`, or `github.event.repository.fork == false`. When uncertain whether the guard applies to every scheduled job, do not flag.

**Refactoring expectation.** Add an `if:` guard restricting scheduled execution to the source repository or owner.

**Expected constructs (2).** `jobs.<job_id>.if`, `jobs.<job_id>.steps[*].if`.

**Safety.** Guard *every* scheduled job: a workflow-level guard that misses one job leaves the smell. Pick the right context value (`github.repository` / `github.repository_owner`).

### 10. UnsafeArtifactUploadFork: *Avoid uploading workflow artifacts on forks*

> Original catalog name: *Avoid uploading artifacts on forks*.

**Detection criterion.** An artifact-upload step exhibits this smell only if ALL conditions are true: (1) the enclosing workflow is triggered by `pull_request` or `pull_request_target`; (2) the step uploads retained GitHub Actions workflow artifacts, i.e., artifacts stored by GitHub Actions as part of the workflow run, using `actions/upload-artifact`, `actions/upload-pages-artifact`, or a wrapper explicitly compatible with `actions/upload-artifact`; and (3) neither the artifact-upload step nor its enclosing job has an effective condition preventing execution for forked pull requests. Effective fork guards include conditions such as `github.event.pull_request.head.repo.fork == false`, `github.event.repository.fork == false`, or equivalent checks that clearly prevent artifact upload when the workflow is running for a fork. Do not flag artifact-upload steps in workflows that are not triggered by `pull_request` or `pull_request_target`. Do not flag coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or files uploaded to external services, registries, cloud storage, or releases, because these are not GitHub Actions workflow artifacts. Do not flag steps that only move, copy, print, cache, download, or pass files within the same runner, job, or workflow without storing them as retained GitHub Actions workflow artifacts. Do not flag artifact-upload steps protected by an effective fork-status guard or cases where the trigger, artifact-upload behavior, or fork guard cannot be determined confidently. Flag one occurrence for each artifact-upload step that satisfies all conditions.

**Refactoring expectation.** Add a fork-status guard (`if:`) on the upload step or its enclosing job.

**Expected constructs (2).** `jobs.<job_id>.if`, `jobs.<job_id>.steps[*].if`.

**Safety.** Guard on fork status; do not disable the upload unconditionally (that removes the smell by removing a legitimate behavior).

### 11. MissingUploadCondition: *Use `if` for workflow artifact-upload actions*

> Original catalog name: *Use `if` for upload-artifact action*.

**Detection criterion.** A step exhibits this smell only if ALL conditions are true: (1) it uploads retained GitHub Actions workflow artifacts using `actions/upload-artifact`, `actions/upload-pages-artifact`, or a wrapper explicitly compatible with `actions/upload-artifact`; (2) the artifact-upload step does not define an explicit step-level `if:` condition; and (3) the enclosing job does not define an explicit job-level `if:` condition. Do not flag artifact-upload steps that define their own step-level `if:` condition, artifact-upload steps inside jobs with a job-level `if:` condition, steps that do not create retained GitHub Actions workflow artifacts, or steps whose upload behavior cannot be determined confidently. Do not treat coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or uploads to external services, registries, cloud storage, or GitHub Releases as retained GitHub Actions workflow artifact uploads. This smell concerns only the absence of an explicit condition controlling artifact upload, not whether the condition is sufficiently restrictive. Flag one occurrence for each artifact-upload step that satisfies all conditions.

**Refactoring expectation.** Add an explicit `if:` on the upload step (or its enclosing job).

**Expected constructs (2).** `jobs.<job_id>.if`, `jobs.<job_id>.steps[*].if`.

**Safety.** Any `if:` satisfies the smell, but a condition that always evaluates false disables the upload entirely; keep it meaningful (e.g. `success()`).

### 12. UnsafeDeployOnFork: *Avoid deploying jobs on forks*

**Detection criterion.** A job exhibits this smell only if ALL conditions are true: (1) the workflow can run in a fork-related pull request context, such as `pull_request`, `pull_request_target`, or another PR-based event; (2) the job directly performs an external delivery operation in the current workflow file through a step-level `uses:` action or a `run:` command, such as deployment, release creation, package publishing, container image push, or cloud/provider deployment; and (3) neither the job nor the delivery step has an effective fork guard. Do not treat job names, workflow names, comments, or job-level reusable workflow calls through `jobs.<job_id>.uses` as sufficient evidence of external delivery by themselves. Concrete evidence of external delivery must be visible in the current workflow file, such as `npm publish`, `docker push`, registry login followed by push, cloud deployment commands, release creation actions, deployment actions, deployment environments, or secrets clearly used for delivery. Do not flag reusable-workflow-only jobs. Do not flag ordinary build, test, lint, analysis, packaging-only, artifact-upload-only, image-build-only, ambiguous jobs, workflows that cannot run from fork-related PR contexts, or jobs protected by an effective fork guard. When uncertain, do not flag. Effective fork guards include conditions such as `github.event.pull_request.head.repo.fork == false`, `github.event.pull_request.head.repo.full_name == github.repository`, `github.event.repository.fork == false`, or equivalent fork-status or same-repository PR checks. Flag one occurrence for each job that satisfies all conditions.

**Refactoring expectation.** Add a fork guard (`if:`) on the delivery job or step.

**Expected constructs (2).** `jobs.<job_id>.if`, `jobs.<job_id>.steps[*].if`.

**Safety.** The guard must block fork PRs. Do *not* "fix" it by adding `pull_request_target` with a checkout of untrusted head code; that is a classic privilege-escalation sink.

### 13. MissingConcurrencyGlobal: *Avoid starting new workflow while previous one is running*

**Detection criterion.** A workflow or job exhibits this smell only if ALL conditions are true: (1) it can be triggered repeatedly for the same logical task, such as scheduled automation, deployment, release, publishing, synchronization, cleanup, data processing, or other long-running or side-effectful automation; (2) overlapping runs could cause unnecessary resource consumption, duplicate side effects, conflicting updates, repeated publishing/deployment, or resource contention; and (3) there is no effective workflow-level or job-level `concurrency:` group that prevents simultaneous or overlapping runs for that same task. Do not flag ordinary CI, build, lint, or test workflows unless there is explicit evidence that overlapping runs are harmful, long-running, resource-intensive, or side-effectful. Do not flag workflows or jobs that already define an effective `concurrency:` group for the same task, with or without `cancel-in-progress`, if it prevents simultaneous overlapping execution. When the repeated trigger, overlap risk, or concurrency configuration cannot be determined confidently, do not flag.

**Refactoring expectation.** Add a `concurrency` group that prevents overlapping runs of the same task (`cancel-in-progress` optional here).

**Expected constructs (6).** `concurrency`, `concurrency.group`, `jobs.<job_id>.concurrency*`. Example: `concurrency.group`.

**Safety.** A group that is too broad serializes independent runs; keep it scoped to the one task that must not overlap.

---

### 14. MutableRunnerVersion: *Use fixed version for runs-on argument*

**Detection criterion.** A job exhibits this smell only if ALL conditions are true: (1) the job defines `runs-on` directly in the current workflow file; and (2) the effective `runs-on` value includes a mutable GitHub-hosted runner label ending in `-latest`, such as `ubuntu-latest`, `windows-latest`, or `macos-latest`. Flag one occurrence for each job that satisfies these conditions. If `runs-on` is a list, flag the job if any label includes a mutable GitHub-hosted runner label ending in `-latest`. Do not flag jobs that use only fixed runner labels such as `ubuntu-22.04`, `windows-2022`, or `macos-14`, self-hosted runner labels, reusable workflow calls using `jobs.<job_id>.uses`, missing `runs-on`, or parameterized `runs-on` values that cannot be resolved from the same file. When the runner label cannot be determined confidently, do not flag.

**Refactoring expectation.** Replace the `-latest` label with a fixed runner version (e.g. `ubuntu-22.04`).

**Expected constructs (4).** `jobs.<job_id>.runs-on`, `jobs.<job_id>.runs-on[*]`, `jobs.<job_id>.runs-on.labels`, `jobs.<job_id>.runs-on.labels[*]`. Example: `jobs.<job_id>.runs-on.labels[*]`.

**Safety.** Pin to a runner label that actually exists; a nonexistent/hallucinated label makes the job unschedulable.

### 15. SuboptimalCacheUsage: *Use cache parameter instead of cache option*

**Detection criterion.** A dependency-cache step exhibits this smell only if ALL conditions are true: (1) the step uses `actions/cache`; (2) the enclosing workflow uses a language or runtime setup action with documented built-in dependency caching, such as `actions/setup-node`, `actions/setup-java`, `actions/setup-python`, `actions/setup-go`, or `ruby/setup-ruby`; (3) the `actions/cache` step caches dependencies for the same language, runtime, or package manager handled by that setup action; and (4) the cache target is not clearly used for another purpose, such as build outputs, generated artifacts, compiler caches, Docker layers, tool downloads, or cross-job cache sharing. Flag one occurrence for each dependency-cache step that satisfies these conditions. Do not flag cache steps for unrelated ecosystems, build outputs, generated artifacts, compiler caches, Docker layers, tool downloads, cross-job cache sharing, or setup actions without a clear built-in dependency caching option. Do not flag cases where the setup action, cache target, or ecosystem match cannot be determined confidently. When uncertain, do not flag.

**Refactoring expectation.** Drop the manual `actions/cache` step and enable the setup action's built-in `cache:` parameter.

**Expected constructs (3).** `jobs.<job_id>.steps[*].uses`, `jobs.<job_id>.steps[*].with`, `jobs.<job_id>.steps[*].with.<parameter>`. Example: `jobs.<job_id>.steps[*].with.<parameter>`.

**Safety.** Confirm the setup action supports built-in caching for that ecosystem before removing `actions/cache`. Removing a still-needed cache degrades performance (not correctness), but removing an unrelated (build-output/Docker-layer) cache is a wrong fix.

### 16. MultiCommandStep: *Use single-command steps*

**Detection criterion.** A `run:` step exhibits this smell only if ALL conditions are true: (1) the step contains two or more relevant shell commands in the same `run:` block; (2) at least two of these commands belong to different workflow phases, such as install, build, test, lint, package, deploy, upload, or cleanup; and (3) these phase-related commands could reasonably be executed as separate `run:` steps. Relevant commands are commands that perform meaningful workflow operations. Do not count auxiliary shell constructs or helper commands by themselves, such as `echo`, `cd`, `pwd`, `ls`, `mkdir`, `export`, variable assignments, conditionals, loops, comments, `true`, `false`, or `sleep`, unless they clearly implement a distinct workflow phase. Flag one occurrence for each `run:` step that satisfies these conditions. Do not flag single relevant commands, commands split across lines only for readability, script invocations such as `npm test` or `python script.py`, short condition/fallback commands for one operation, tightly coupled commands that perform one task, or cases where the commands, phases, or separability cannot be determined confidently.

**Refactoring expectation.** Split the multi-phase `run:` step into separate single-purpose steps.

**Expected constructs (4).** `jobs.<job_id>.steps[*].run`, `jobs.<job_id>.steps[*].name`, `jobs.<job_id>.steps[*].shell`, `jobs.<job_id>.steps[*].working-directory`. Example: `jobs.<job_id>.steps[*].name`.

**Safety.** Steps do not share working directory or shell state by default. A naive split that separates a `cd` / `export` from the command that depends on it, or reorders phases, breaks the build. Preserve ordering and per-step context.

### 17. SingleOSExecution: *Run tests on multiple OS's*

**Detection criterion.** A job exhibits this smell only if ALL conditions are true: (1) the job is a test job, identified by test-related job names or commands such as `test`, `pytest`, `npm test`, `yarn test`, `mvn test`, `gradle test`, `go test`, `cargo test`, or equivalent; (2) the job has exactly one effective operating system in its `runs-on` configuration, or equivalent runner configuration; and (3) the job does not define an OS matrix or equivalent strategy with two or more distinct operating system values. Flag one occurrence for each test job that satisfies these conditions. If the job defines a matrix, but the matrix contains only one distinct operating system value, still flag the job because it effectively tests on only one OS. Do not flag non-test jobs, deployment or documentation jobs, jobs that effectively test on two or more operating systems, jobs running inside a fixed container environment, or cases where the job purpose or operating system configuration cannot be determined confidently. When uncertain, do not flag.

**Refactoring expectation.** Add an OS matrix (`strategy.matrix`) with at least 2 operating systems and wire it into `runs-on`.

**Expected constructs (6).** `jobs.<job_id>.strategy.matrix`, `jobs.<job_id>.strategy.matrix.<matrix_var>`, `jobs.<job_id>.strategy.matrix.include`, `jobs.<job_id>.strategy.matrix.include[*].<matrix_var>`, `jobs.<job_id>.runs-on`, `jobs.<job_id>.runs-on[*]`. Example: `jobs.<job_id>.strategy.matrix.<matrix_var>`.

**Safety.** The matrix variable must actually feed `runs-on`; a matrix that is declared but not referenced changes nothing (smell not removed).

### 18. UnpinnedDependencies: *Specify package versions*

**Detection criterion.** A `run:` step exhibits this smell only if ALL conditions are true: (1) it installs external packages, tools, or dependencies using commands such as `apt install`, `apt-get install`, `brew install`, `pip install`, `pip install -r`, `npm install -g`, `gem install`, `cargo install`, or similar; (2) at least one installed package/tool is not explicitly pinned to a fixed version, commit SHA, digest, checksum-verified artifact, immutable release URL, or fixed version constraint; and (3) the installation can affect build, test, analysis, deployment, or release behavior. Treat dependency files such as `requirements.txt` as unpinned unless their contents are available and every dependency is pinned to an exact version. Do not flag package index updates without installation, installs from pinned commits/digests/immutable releases, installs with exact version constraints, or cases governed by verified lockfiles such as `package-lock.json`, `yarn.lock`, `poetry.lock`, `Pipfile.lock`, `Gemfile.lock`, or equivalent. When a referenced dependency file cannot be inspected, flag it as unpinned.

**Refactoring expectation.** Pin each installed package to a fixed version (or move to a verified lockfile).

**Expected constructs (1).** `jobs.<job_id>.steps[*].run`.

**Safety.** Pin to versions that exist and are compatible; do not pin index-update-only commands (`apt-get update`), which install nothing.

### 19. SingleLangVersionCI: *Run CI on multiple language versions*

**Detection criterion.** A CI job exhibits this smell only if ALL conditions are true: (1) the job performs CI validation, such as testing, building, linting, static analysis, or integration checks; (2) the job configures exactly one explicit project language or runtime version in the workflow file; and (3) the job does not define a matrix or equivalent strategy that runs the same CI validation on two or more distinct project language/runtime versions. This smell concerns only project language or runtime versions, not operating systems. Do not treat `runs-on`, OS labels such as `ubuntu-latest`, `windows-latest`, or `macos-latest`, or OS matrices as runtime-version coverage. Treat versions configured through project runtime setup actions, such as `actions/setup-node` with `node-version`, `actions/setup-python` with `python-version`, `actions/setup-java` with `java-version`, `actions/setup-go` with `go-version`, `ruby/setup-ruby` with `ruby-version`, or equivalent setup actions, as explicit runtime versions. If a matrix exists but contains only one distinct runtime version, still flag the job. Do not flag non-CI jobs, deployment-only jobs, documentation jobs, jobs already testing two or more distinct runtime versions, jobs without an explicit project runtime version in the workflow file, jobs that configure only auxiliary tool versions, or jobs whose CI purpose or runtime-version configuration cannot be determined confidently. When uncertain, do not flag. Flag one occurrence for each CI job that satisfies all conditions.

**Refactoring expectation.** Add a runtime-version matrix (`strategy.matrix`) with at least 2 versions and wire it into the setup action's version input.

**Expected constructs (5).** `jobs.<job_id>.strategy.matrix`, `jobs.<job_id>.strategy.matrix.<matrix_var>`, `jobs.<job_id>.strategy.matrix.include`, `jobs.<job_id>.strategy.matrix.include[*].<matrix_var>`, `jobs.<job_id>.steps[*].with.<parameter>`. Example: `jobs.<job_id>.strategy.matrix.include[*].<matrix_var>`.

**Safety.** The matrix variable must feed the setup action's version input (e.g. `node-version: ${{ matrix.node }}`); an unreferenced matrix var is a no-op and does not remove the smell.
