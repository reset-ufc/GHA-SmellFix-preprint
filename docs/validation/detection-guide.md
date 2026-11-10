# Detection annotation guide

The 19 GitHub Actions configuration smells this study labels, with the information an annotator needs to decide *present* vs *absent* for each `(workflow, smell)` pair.

### How to read this guide

Each smell provides a **Problem** and a **Solution**, followed by the **Definition: flagging criterion** used for labeling. When applicable, a smell also includes **How to identify**, **Review notes**, or a **Caution / Gotcha**. The annotated examples use the convention:

> **Problem** and **Solution** come from the original catalog by Khatami et al. (the source from which these smells were selected), and are shown only to reinforce each smell's definition and motivation. The **Definition** below each smell is the flagging rule refined for and used by this study.

- **With smell**: the workflow snippet exhibits the smell.
- **Without smell**: the snippet is clean (serves as a counterexample).


### 1. Define permissions for workflows with external actions

- **Problem:** Overly permissive access increases security risks.
- **Solution:** Specify minimal permissions.

**Definition: flagging criterion**

> A job exhibits this smell only if ALL conditions are true: (1) it contains at least one step-level `uses:` on a `job`; (2) at least one reference is a GitHub action in the form `owner/repo@ref` or `owner/repo/path@ref`; (3) the `owner/repo` portion differs from the provided `current_repository`; and (4) the job has no effective `permissions:` configuration, meaning there is neither a workflow-level nor job-level `permissions:` block. Treat relative-path references, such as `uses: ./.github/actions/setup`, as local actions. Do not flag local actions, same-repository action references, job-level `uses:` references to reusable workflows, Docker image references such as `uses: docker://alpine:3.19`, unparsable references, or jobs with any applicable `permissions:` block, even if permissions are broad. Flag one occurrence per job, regardless of how many external step-level actions it contains.

**Review notes**

- Do not flag workflows that call reusable workflows: without `steps` they have no action call, so there is no smell.
- Do not flag `step.uses` pointing to a LOCAL ACTION by relative path (`./.github/actions/...`) or to an action from the SAME REPOSITORY as the original.

**Examples**

- **With smell**: Job uses an external action (`actions/checkout`) and there is no `permissions`, neither at the job nor at the top level.

  ```yaml
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
  ```

- **Without smell**: Same external action, but with `permissions` declared at the top of the workflow.

  ```yaml
  permissions:
    contents: read
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
  ```

- **Without smell**: `permissions` present on the job counts; even broad (`write-all`) is not a smell.

  ```yaml
  jobs:
    build:
      runs-on: ubuntu-latest
      permissions: write-all
      steps:
        - uses: actions/checkout@v4
  ```

- **Without smell**: A LOCAL action by relative path is not an external action.

  ```yaml
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: ./.github/actions/setup
  ```

- **Without smell**: Reusable workflow (job-level `uses`, no `steps`) does not call an action.

  ```yaml
  jobs:
    call:
      uses: ./.github/workflows/ci.yml
  ```


### 2. Use commit hash instead of tags for action versions

- **Problem:** Tags can be modified, leading to inconsistent behavior.
- **Solution:** `uses: actions/checkout@<commit-sha>`

**Definition: flagging criterion**

> A step-level `uses:` reference exhibits this smell only if ALL conditions are true: (1) the reference appears under `jobs.<job_id>.steps[*].uses`; (2) the `uses:` value is a GitHub action reference in the form `owner/repo@ref` or `owner/repo/path@ref`; and (3) `ref` is not exactly a 40-character hexadecimal commit SHA. Flag branch names, tags, major-version aliases, and floating references such as `@main`, `@master`, `@develop`, `@v1`, `@v2`, `@latest`, or any other non-40-character hexadecimal reference. This applies to external actions and to same-repository actions when they are referenced through the explicit `owner/repo@ref` or `owner/repo/path@ref` form. Do not flag local actions referenced by relative path, such as `uses: ./.github/actions/setup`, because they do not use an explicit `@ref`. Do not flag actions pinned to a full 40-character hexadecimal commit SHA, `uses:` values without `@ref`, non-GitHub Docker image references such as `uses: docker://alpine:3.19`, job-level `uses:` references to reusable workflows, including local workflows such as `jobs.<job_id>.uses: ./.github/workflows/ci.yml`, or references whose value cannot be parsed confidently. Flag one occurrence for each step-level `uses:` reference that satisfies all conditions.

**Review notes**

- Do not flag workflows that call reusable workflows (no `step.uses`).
- Do not flag `step.uses` of a LOCAL ACTION by relative path (`./.github/actions/...`).
- Do flag normally when the action, even if from the same repository, is referenced in the `owner/repo@ref` form with a tag/branch instead of a SHA.

**Examples**

- **With smell**: External `uses` pinned to a tag/branch instead of a SHA.

  ```yaml
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@main
  ```

- **Without smell**: Pinned to a 40-character commit SHA.

  ```yaml
  steps:
    - uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11
  ```

- **Without smell**: Local action by relative path (no `@ref`).

  ```yaml
  steps:
    - uses: ./.github/actions/setup
  ```

- **Without smell**: Reusable workflow (job-level `uses`, no `step.uses`).

  ```yaml
  jobs:
    call:
      uses: ./.github/workflows/release.yml
  ```

- **With smell**: Same repo, but in the `owner/repo/path@tag` form → counts as a smell.

  ```yaml
  steps:
    - uses: my-org/my-repo/.github/actions/build@v1
  ```


### 3. Set permissions for GitHub Token

- **Problem:** Default token has overly permissive access.
- **Solution:** Set permissions under the permissions key.
- **How to identify:** When the workflow uses `secrets.GITHUB_TOKEN`, the job must state what that token can do. Look for the word `GITHUB_TOKEN` and confirm that `permissions:` exist in the same job.

**Definition: flagging criterion**

> A job exhibits this smell only if ALL conditions are true: (1) the job uses the GitHub-provided token explicitly through `secrets.GITHUB_TOKEN` or `github.token`, or uses an action/command that clearly performs authenticated GitHub operations such as creating comments, labels, releases, deployments, packages, statuses, pull request updates, issue updates, or repository writes; (2) the job has no effective `permissions:` configuration, either from a workflow-level `permissions:` block or a job-level `permissions:` block. Do not flag jobs with an applicable workflow-level or job-level `permissions:` block, even if the permissions are broad. Do not flag jobs only because they run on GitHub Actions, use `actions/checkout`, or perform ordinary build/test steps. This smell concerns only the absence of explicit token permission configuration, not whether the configured permissions follow least privilege. When token use is uncertain, do not flag.

**Caution / Gotcha**

> This smell is valid only if the token is referenced in the steps without a defined restriction. If the token is not used in the file, there is no smell.



### 4. Prevent running issue/PR actions on forks

- **Problem:** Actions fail due to lack of permissions.
- **Solution:** Add condition checking repository owner.
- **How to identify:** Workflows triggered by `pull_request` or `issues` must check that they run only in the original repository (not in forks). Look for an `if:` condition that checks `github.repository_owner` or `github.repository`.

**Definition: flagging criterion**

> A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by an issue- or pull-request-related event that may involve forked repositories, such as `pull_request`, `pull_request_target`, `issue_comment`, `pull_request_review`, or `pull_request_review_comment`; (2) at least one job or relevant step performs an issue or pull request action, such as commenting, labeling, approving, updating statuses, or modifying issue/PR metadata; and (3) that job or step has no effective fork guard. Effective fork guards must clearly prevent the issue/PR action from running for forked pull requests, such as `github.event.pull_request.head.repo.fork == false`, `github.event.pull_request.head.repo.full_name == github.repository`, or equivalent same-repository PR checks. For `issue_comment`, the guard must also ensure the comment belongs to a pull request before checking the associated PR fork status. Do not treat repository-identity checks alone, label-based checks, branch checks, actor checks, event-type checks, or comment-content checks as fork guards unless combined with an explicit fork-status or same-repository PR check. Do not flag workflows that are not triggered by issue/PR-related events, do not perform issue/PR actions, are protected by effective fork guards, or cannot be determined confidently. Flag one occurrence for each workflow that satisfies all conditions.

**Caution / Gotcha**

> Verify that the workflow is actually triggered by `pull_request` or `issue`. There are also other ways to check that the execution environment is not a fork.


### 5. Avoid jobs without timeouts

- **Problem:** Can run indefinitely, wasting resources and blocking workflows.
- **Solution:** Set timeout for jobs.

**Definition: flagging criterion**

> A job exhibits this smell only if ALL conditions are true: (1) it is a job entry defined in the current workflow file; and (2) it does not define `timeout-minutes` at the job level. Flag one occurrence for each job that satisfies these conditions, even if one or more steps define their own `timeout-minutes`, and even if the job has no `steps` block. Step-level `timeout-minutes` does not count as a job-level timeout and does not prevent this smell. Do not flag jobs that explicitly define `timeout-minutes` at the job level or jobs whose structure or timeout configuration cannot be determined confidently. This rule checks only whether `timeout-minutes` is present at the job level in the current workflow file; it does not inspect called reusable workflows or their internal timeout configuration.

**Review notes**

- `timeout-minutes` on individual steps does NOT replace the job's `timeout-minutes`. Each job MUST have its own timeout at the job level.
- Jobs without `steps:` (e.g., jobs that call reusable workflows via `uses:`) must also be flagged if they lack `timeout-minutes` at the job level.
- Do not inspect called reusable workflows; analyze only the current workflow file.
- Each job without `timeout-minutes` at the job level counts as one occurrence.

**Examples**

- **With smell**: Job without `timeout-minutes` may run up to the default 6h limit.

  ```yaml
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - run: make
  ```

- **With smell**: The step has `timeout-minutes` but the job does not; step-level does not replace job-level.

  ```yaml
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - run: npm test
          timeout-minutes: 10
  ```

- **With smell**: Job calls a reusable workflow without `timeout-minutes` at the job level.

  ```yaml
  jobs:
    deploy:
      uses: ./.github/workflows/deploy.yml
  ```

- **With smell**: Two jobs, neither with `timeout-minutes` (two occurrences).

  ```yaml
  jobs:
    lint:
      runs-on: ubuntu-latest
      steps:
        - run: npm run lint
    test:
      runs-on: ubuntu-latest
      steps:
        - run: npm test
  ```

- **Without smell**: Job with `timeout-minutes` at the job level is protected.

  ```yaml
  jobs:
    build:
      runs-on: ubuntu-latest
      timeout-minutes: 15
      steps:
        - run: make
  ```

- **Without smell**: Job with a reusable workflow and `timeout-minutes` is protected.

  ```yaml
  jobs:
    deploy:
      uses: ./.github/workflows/deploy.yml
      timeout-minutes: 30
  ```


### 6. Stop running workflows when there is a newer commit in PR

- **Problem:** Inefficient resource usage and inconsistent results.
- **Solution:** Use concurrency to cancel in-progress runs.
- **How to identify:** When a PR receives a new commit, the previous run keeps running for nothing. Workflows with `on: pull_request` need a `concurrency:` block that cancels the old run. Pay attention to whether the workflow is actually triggered by `pull_request`.

**Definition: flagging criterion**

> A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `pull_request` or `pull_request_target`; (2) it does not define an effective workflow-level `concurrency:` configuration with `cancel-in-progress: true`; and (3) at least one job that can run on the PR trigger does not define an effective job-level `concurrency:` configuration with `cancel-in-progress: true`. Flag one occurrence for each workflow where newer commits to a pull request may start additional workflow runs without canceling older in-progress runs for that PR. Do not flag workflows without `pull_request` or `pull_request_target` triggers. Do not flag workflows with workflow-level `concurrency.cancel-in-progress: true`, or workflows where every job that can run on the PR trigger has job-level `concurrency.cancel-in-progress: true`. For this smell, only check whether an applicable `concurrency:` block with `cancel-in-progress: true` exists. Do not evaluate whether the `concurrency.group` expression is optimal, pull-request-specific, branch-specific, ref-specific, workflow-specific, or otherwise well designed. Do not flag configurations whose `concurrency:` or `cancel-in-progress` value cannot be confidently interpreted. When uncertain, do not flag.


### 7. Stop running workflows when there is a newer commit in branch

- **Problem:** Inefficient resource usage and inconsistent results.
- **Solution:** Use concurrency to cancel in-progress runs.

**Definition: flagging criterion**

> A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `push` events; (2) it does not define an effective workflow-level `concurrency:` configuration with `cancel-in-progress: true`; and (3) at least one job that can run on branch `push` events does not define an effective job-level `concurrency:` configuration with `cancel-in-progress: true`. Flag one occurrence for each workflow where newer commits pushed to a branch may start additional workflow runs without canceling older in-progress runs for that branch. Do not flag workflows that are not triggered by `push`, workflows triggered only by tag pushes, workflows with workflow-level `concurrency.cancel-in-progress: true`, or workflows where every job that can run on branch `push` events has job-level `concurrency.cancel-in-progress: true`. For this smell, only check whether an applicable `concurrency:` block with `cancel-in-progress: true` exists. Do not evaluate whether the `concurrency.group` expression is optimal, branch-specific, workflow-specific, or otherwise well designed. Do not flag configurations whose `concurrency:` or `cancel-in-progress` value cannot be confidently interpreted. When uncertain, do not flag.

**Review notes**

- GOTCHA: Workflows that have `on: push` but filter ONLY by tags (e.g., `push: tags: [...]`) must NOT be flagged; tags do not generate sequential commits on the same branch that would need cancellation.
- If the workflow has `on: push` with `branches:` or no filter at all, then it is a candidate for this smell.
- It is enough for `concurrency:` with `cancel-in-progress: true` to exist; do not evaluate whether the `group` is ideal or branch-specific.
- Concurrency may be at the workflow OR job level. If ALL jobs that run on branch push have `concurrency` with `cancel-in-progress: true`, do not flag.

**Examples**

- **With smell**: `on: push` without `concurrency:`; old runs are never canceled.

  ```yaml
  on: [push]
  jobs:
    ci:
      runs-on: ubuntu-latest
      steps:
        - run: make
  ```

- **With smell**: `concurrency:` exists but without `cancel-in-progress: true`; does not cancel old runs.

  ```yaml
  on: [push]
  concurrency:
    group: ci-${{ github.ref }}
  jobs:
    ci:
      runs-on: ubuntu-latest
      steps:
        - run: make
  ```

- **With smell**: `on: push` with `branches:` but without `concurrency:`; pushes to the branch accumulate runs.

  ```yaml
  on:
    push:
      branches: [main, develop]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - run: npm test
  ```

- **Without smell**: Workflow-level `concurrency` with `cancel-in-progress: true` is protected.

  ```yaml
  on: [push]
  concurrency:
    group: ci-${{ github.ref }}
    cancel-in-progress: true
  jobs:
    ci:
      runs-on: ubuntu-latest
      steps:
        - run: make
  ```

- **Without smell**: Job-level `concurrency` with `cancel-in-progress: true` on all jobs is protected.

  ```yaml
  on: [push]
  jobs:
    ci:
      concurrency:
        group: ci-${{ github.ref }}
        cancel-in-progress: true
      runs-on: ubuntu-latest
      steps:
        - run: make
  ```

- **Without smell**: GOTCHA: `push` filters ONLY by tags; does not generate sequential runs on a branch.

  ```yaml
  name: Publish
  on:
    pull_request:
      branches: [master]
    push:
      tags: ['[A-z]+-v[0-9]+.[0-9]+.[0-9]+*']
  jobs:
    publish:
      if: ${{ github.repository_owner == 'google' }}
      uses: dart-lang/ecosystem/.github/workflows/publish.yaml@main
  ```

- **Without smell**: Workflow does not have `on: push` (out of scope for this smell).

  ```yaml
  on: [pull_request]
  jobs:
    ci:
      runs-on: ubuntu-latest
      steps:
        - run: npm test
  ```


### 8. Avoid running CI actions when no source code has changed

- **Problem:** Unnecessary resource usage when irrelevant files change.
- **Solution:** Specify trigger files using paths or paths-ignore.

**Definition: flagging criterion**

> A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `push` or `pull_request`; (2) its main purpose is CI validation, such as build, test, lint, static analysis, packaging checks, or dependency checks; and (3) at least one relevant `push` or `pull_request` trigger does not define `paths` or `paths-ignore`. Flag one occurrence for each workflow whose relevant CI trigger lacks `paths` or `paths-ignore`. Do not flag workflows where every relevant `push` or `pull_request` trigger defines `paths` or `paths-ignore`, regardless of whether the filter is complete, precise, or optimal. Do not flag workflows whose main purpose is documentation, release, deployment, notification, scheduled maintenance, formatting-only automation, whole-repository automation, or cases where the workflow purpose cannot be determined confidently. When the trigger, workflow purpose, or path-filter configuration cannot be determined confidently, do not flag.

**Review notes**

- Workflows triggered by `pull_request` are in scope, not only `push`. Do NOT confuse with `pull_request_target`; that one must not be considered a smell.
- Do not flag when the workflow does ONLY deploy, notification, scheduled maintenance, formatting-only automation, or whole-repository automation.

**Examples**

- **With smell**: CI on `push` without `paths`/`paths-ignore`.

  ```yaml
  name: CI
  on:
    push:
      branches: [main]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - run: pytest
  ```

- **With smell**: CI on `pull_request` without a filter is now also in scope.

  ```yaml
  name: CI
  on: [pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - run: pytest
  ```

- **Without smell**: CI trigger with `paths-ignore` defined (a filter present is enough).

  ```yaml
  name: CI
  on:
    push:
      branches: [main]
      paths-ignore:
        - 'docs/**'
        - '**.md'
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - run: pytest
  ```

- **Without smell**: `pull_request_target` is not considered a smell, even without a filter.

  ```yaml
  on: [pull_request_target]
  jobs:
    label:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/labeler@v5
  ```

- **Without smell**: Purpose is deploy (non-CI); out of scope, even without a filter.

  ```yaml
  name: Deploy
  on:
    push:
      branches: [main]
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - run: ./deploy.sh
  ```


### 9. Avoid executing scheduled workflows on forks

- **Problem:** Inefficient resource usage for inactive forks.
- **Solution:** Add condition checking repository owner.
- **How to identify:** Scheduled workflows (`on: schedule`) can run on forks that re-enabled the Actions tab. Each such job must have an `if:` checking the repository owner.

**Definition: flagging criterion**

> A workflow exhibits this smell only if ALL conditions are true: (1) it is triggered by `schedule`; (2) at least one scheduled job can run without a condition restricting execution to the source repository or repository owner; and (3) there is no effective workflow-level or job-level `if:` condition using values such as `github.repository`, `github.repository_owner` or `github.event.repository.fork` to restrict scheduled execution. Do not flag workflows that are not triggered by `schedule`, or scheduled workflows where every scheduled job is protected by a repository or owner check, such as `github.repository == 'owner/repo'`, `github.repository_owner == 'owner'` or `github.event.repository.fork == false`. When uncertain whether the guard applies to every scheduled job, do not flag.

**Caution / Gotcha**

> Verify whether the workflow has `schedule`; if it does not, there is no smell. If it has `schedule` and checks for a fork, then there is no smell, even if the way it checks for a fork is not the same as in the example.


### 10. Avoid uploading workflow artifacts on forks

- **Problem:** Inefficient resource usage.
- **Solution:** Add condition checking repository owner.

**Definition: flagging criterion**

> An artifact-upload step exhibits this smell only if ALL conditions are true: (1) the enclosing workflow is triggered by `pull_request` or `pull_request_target`; (2) the step uploads retained GitHub Actions workflow artifacts, i.e., artifacts stored by GitHub Actions as part of the workflow run, using `actions/upload-artifact`, `actions/upload-pages-artifact`, or a wrapper explicitly compatible with `actions/upload-artifact`; and (3) neither the artifact-upload step nor its enclosing job has an effective condition preventing execution for forked pull requests. Effective fork guards include conditions such as `github.event.pull_request.head.repo.fork == false`, `github.event.repository.fork == false`, or equivalent checks that clearly prevent artifact upload when the workflow is running for a fork. Do not flag artifact-upload steps in workflows that are not triggered by `pull_request` or `pull_request_target`. Do not flag coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or files uploaded to external services, registries, cloud storage, or releases, because these are not GitHub Actions workflow artifacts. Do not flag steps that only move, copy, print, cache, download, or pass files within the same runner, job, or workflow without storing them as retained GitHub Actions workflow artifacts. Do not flag artifact-upload steps protected by an effective fork-status guard or cases where the trigger, artifact-upload behavior, or fork guard cannot be determined confidently. Flag one occurrence for each artifact-upload step that satisfies all conditions.

**Review notes**

- We should consider as a smell only those triggered by `pull_request`, `pull_request_target`.
- Consider as an artifact-upload mechanism only `actions/upload-artifact` or clearly equivalent actions that create retained GitHub Actions artifacts for the current workflow run.
- Do not treat coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or uploads to external services, registries, cloud storage, or GitHub Releases as equivalent artifact uploads, because they do not create the GitHub Actions workflow artifacts targeted by this smell.
- When in doubt about this smell, ask in the validation group.

**Examples**

- **With smell**: Workflow triggered by `pull_request` with `upload-artifact` and no fork guard.

  ```yaml
  on: [pull_request]
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - run: make
        - uses: actions/upload-artifact@v4
          with:
            name: build
            path: dist/
  ```

- **With smell**: `upload-pages-artifact` in a `pull_request_target` workflow without a guard is also a smell.

  ```yaml
  on: [pull_request_target]
  jobs:
    pages:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/upload-pages-artifact@v3
          with:
            path: public/
  ```

- **Without smell**: Step with a fork guard (`head.repo.fork == false`) is protected.

  ```yaml
  on: [pull_request]
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/upload-artifact@v4
          if: github.event.pull_request.head.repo.fork == false
          with:
            name: build
            path: dist/
  ```

- **Without smell**: Job-level `if:` with a fork guard protects all steps in the job.

  ```yaml
  on: [pull_request]
  jobs:
    build:
      if: github.event.pull_request.head.repo.full_name == github.repository
      runs-on: ubuntu-latest
      steps:
        - uses: actions/upload-artifact@v4
          with:
            name: build
            path: dist/
  ```

- **Without smell**: Workflow triggered by `push` (not a PR) is out of scope.

  ```yaml
  on: [push]
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/upload-artifact@v4
          with:
            name: build
            path: dist/
  ```

- **Without smell**: Coverage upload (`codecov`) is not a retained GitHub Actions artifact.

  ```yaml
  on: [pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: codecov/codecov-action@v4
  ```


### 11. Use `if` for upload-artifact action

- **Problem:** Unnecessary uploads waste resources and storage.
- **Solution:** Add condition to run only when needed.

**Definition: flagging criterion**

> A step exhibits this smell only if ALL conditions are true: (1) it uploads retained GitHub Actions workflow artifacts using `actions/upload-artifact`, `actions/upload-pages-artifact`, or a wrapper explicitly compatible with `actions/upload-artifact`; (2) the artifact-upload step does not define an explicit step-level `if:` condition; and (3) the enclosing job does not define an explicit job-level `if:` condition. Do not flag artifact-upload steps that define their own step-level `if:` condition, artifact-upload steps inside jobs with a job-level `if:` condition, steps that do not create retained GitHub Actions workflow artifacts, or steps whose upload behavior cannot be determined confidently. Do not treat coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or uploads to external services, registries, cloud storage, or GitHub Releases as retained GitHub Actions workflow artifact uploads. This smell concerns only the absence of an explicit condition controlling artifact upload, not whether the condition is sufficiently restrictive. Flag one occurrence for each artifact-upload step that satisfies all conditions.

**Review notes**

- Consider as an artifact-upload mechanism only `actions/upload-artifact` or clearly equivalent actions that create retained GitHub Actions artifacts for the current workflow run.
- Do not treat coverage uploads, release assets, packages, container images, cache entries, logs, downloads, or uploads to external services, registries, cloud storage, or GitHub Releases as equivalent artifact uploads, because they do not create the GitHub Actions workflow artifacts targeted by this smell.
- When in doubt about this smell, ask in the validation group.

**Examples**

- **With smell**: `actions/upload-artifact` without `if:` on the step or the job always runs.

  ```yaml
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - run: make
        - uses: actions/upload-artifact@v4
          with:
            name: build
            path: dist/
  ```

- **With smell**: `actions/upload-pages-artifact` without a condition is also a retained artifact.

  ```yaml
  jobs:
    pages:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/upload-pages-artifact@v3
          with:
            path: public/
  ```

- **Without smell**: Step with an explicit `if:`. The condition itself does not matter, it just needs to exist.

  ```yaml
  steps:
    - uses: actions/upload-artifact@v4
      if: success()
      with:
        name: build
        path: dist/
  ```

- **Without smell**: Job with `if:` protects all steps, including the upload.

  ```yaml
  jobs:
    build:
      if: github.ref == 'refs/heads/main'
      runs-on: ubuntu-latest
      steps:
        - uses: actions/upload-artifact@v4
          with:
            name: build
            path: dist/
  ```

- **Without smell**: Coverage upload (`codecov`) is not a retained GitHub Actions artifact.

  ```yaml
  steps:
    - uses: codecov/codecov-action@v4
  ```

- **Without smell**: Upload to an external registry (Docker Hub) is not a GitHub Actions artifact.

  ```yaml
  steps:
    - run: docker push myorg/myimage:latest
  ```


### 12. Avoid deploying jobs on forks

- **Problem:** Inefficient resource usage due to unnecessary deployments.
- **Solution:** Add condition checking repository owner.

**Definition: flagging criterion**

> A job exhibits this smell only if ALL conditions are true: (1) the workflow can run in a fork-related pull request context, such as `pull_request`, `pull_request_target`, or another PR-based event; (2) the job directly performs an external delivery operation in the current workflow file through a step-level `uses:` action or a `run:` command, such as deployment, release creation, package publishing, container image push, or cloud/provider deployment; and (3) neither the job nor the delivery step has an effective fork guard. Do not treat job names, workflow names, comments, or job-level reusable workflow calls through `jobs.<job_id>.uses` as sufficient evidence of external delivery by themselves. Concrete evidence of external delivery must be visible in the current workflow file, such as `npm publish`, `docker push`, registry login followed by push, cloud deployment commands, release creation actions, deployment actions, deployment environments, or secrets clearly used for delivery. Do not flag reusable-workflow-only jobs. Do not flag ordinary build, test, lint, analysis, packaging-only, artifact-upload-only, image-build-only, ambiguous jobs, workflows that cannot run from fork-related PR contexts, or jobs protected by an effective fork guard. When uncertain, do not flag. Effective fork guards include conditions such as `github.event.pull_request.head.repo.fork == false`, `github.event.pull_request.head.repo.full_name == github.repository`, `github.event.repository.fork == false`, or equivalent fork-status or same-repository PR checks. Flag one occurrence for each job that satisfies all conditions.

**Review notes**

- Restricted to workflows triggered by `pull_request`, `pull_request_target`, or an equivalent PR event. Do not flag workflows that cannot run in a fork/PR context.
- You must determine whether there is any `run` command or `uses` action that actually performs deploy/external delivery. Job names, workflow names, or comments are NOT sufficient evidence.
- Do not flag jobs that only do build, test, lint, analysis, packaging, artifact upload, or image build without push.
- Jobs that call reusable workflows (`jobs.<job_id>.uses` without `steps`) do not count as delivery evidence.

**Examples**

- **With smell**: Job runs `npm publish` in a `pull_request` workflow without a fork guard.

  ```yaml
  on: [pull_request]
  jobs:
    publish:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: npm publish
  ```

- **With smell**: `docker push` in a `pull_request_target` workflow without a guard is external delivery.

  ```yaml
  on: [pull_request_target]
  jobs:
    push-image:
      runs-on: ubuntu-latest
      steps:
        - run: docker login -u $USER -p $TOKEN
        - run: docker push myorg/myapp:latest
  ```

- **With smell**: Deploy action (`aws-actions/amazon-ecs-deploy-task-definition`) without a fork guard.

  ```yaml
  on: [pull_request]
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - uses: aws-actions/amazon-ecs-deploy-task-definition@v1
          with:
            task-definition: task-def.json
  ```

- **Without smell**: Job with a fork guard (`head.repo.fork == false`) is protected.

  ```yaml
  on: [pull_request]
  jobs:
    deploy:
      if: github.event.pull_request.head.repo.fork == false
      runs-on: ubuntu-latest
      steps:
        - run: ./deploy.sh
  ```

- **Without smell**: Workflow triggered by `push` (not a PR) is out of scope.

  ```yaml
  on: [push]
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - run: ./deploy.sh
  ```

- **Without smell**: Job only does build and test, not external delivery, even without a guard.

  ```yaml
  on: [pull_request]
  jobs:
    ci:
      runs-on: ubuntu-latest
      steps:
        - run: npm ci
        - run: npm test
  ```

- **Without smell**: Job calls a reusable workflow (`jobs.deploy.uses`). No `steps`, does not count.

  ```yaml
  on: [pull_request]
  jobs:
    deploy:
      uses: ./.github/workflows/deploy.yml
  ```


### 13. Avoid starting new workflow while previous one is running

- **Problem:** Inefficient resource usage and inconsistent states.
- **Solution:** Use concurrency groups to ensure only one runs at a time.
- **How to identify:** You must analyze whether the workflow CAN/SHOULD overlap if the previous run is slow. Some workflows, such as test workflows, cause no cost problems when running in parallel; however, workflows such as "release" or "scheduled" ones normally should have `concurrency:` to ensure only one runs at a time. This smell depends heavily on the workflow context; your job is mainly to try to understand whether the workflow will cause "inefficient resource consumption" if it runs without canceling the previous execution.

**Definition: flagging criterion**

> A workflow or job exhibits this smell only if ALL conditions are true: (1) it can be triggered repeatedly for the same logical task, such as scheduled automation, deployment, release, publishing, synchronization, cleanup, data processing, or other long-running or side-effectful automation; (2) overlapping runs could cause unnecessary resource consumption, duplicate side effects, conflicting updates, repeated publishing/deployment, or resource contention; and (3) there is no effective workflow-level or job-level `concurrency:` group that prevents simultaneous or overlapping runs for that same task. Do not flag ordinary CI, build, lint, or test workflows unless there is explicit evidence that overlapping runs are harmful, long-running, resource-intensive, or side-effectful. Do not flag workflows or jobs that already define an effective `concurrency:` group for the same task, with or without `cancel-in-progress`, if it prevents simultaneous overlapping execution. When the repeated trigger, overlap risk, or concurrency configuration cannot be determined confidently, do not flag.



### 14. Use fixed version for runs-on argument

- **Problem:** Environment changes can lead to unexpected behavior.
- **Solution:** Specify exact version for runs-on.
- **How to identify:** Look for lines with `runs-on: ...-latest` (e.g., `ubuntu-latest`, `macos-latest`). The `-latest` suffix changes when GitHub updates the runner, breaking reproducibility. Anything in `runs-on` that uses versions that are not "fixed" is categorized as a smell.

**Definition: flagging criterion**

> A job exhibits this smell only if ALL conditions are true: (1) the job defines `runs-on` directly in the current workflow file; and (2) the effective `runs-on` value includes a mutable GitHub-hosted runner label ending in `-latest`, such as `ubuntu-latest`, `windows-latest`, or `macos-latest`. Flag one occurrence for each job that satisfies these conditions. If `runs-on` is a list, flag the job if any label includes a mutable GitHub-hosted runner label ending in `-latest`. Do not flag jobs that use only fixed runner labels such as `ubuntu-22.04`, `windows-2022`, or `macos-14`, self-hosted runner labels, reusable workflow calls using `jobs.<job_id>.uses`, missing `runs-on`, or parameterized `runs-on` values that cannot be resolved from the same file. When the runner label cannot be determined confidently, do not flag.


### 15. Use cache parameter instead of cache option

- **Problem:** Cache options increase workflow complexity and misconfiguration risk.
- **Solution:** Update workflows to use cache parameter.

**Definition: flagging criterion**

> A dependency-cache step exhibits this smell only if ALL conditions are true: (1) the step uses `actions/cache`; (2) the enclosing workflow uses a language or runtime setup action with documented built-in dependency caching, such as `actions/setup-node`, `actions/setup-java`, `actions/setup-python`, `actions/setup-go`, or `ruby/setup-ruby`; (3) the `actions/cache` step caches dependencies for the same language, runtime, or package manager handled by that setup action; and (4) the cache target is not clearly used for another purpose, such as build outputs, generated artifacts, compiler caches, Docker layers, tool downloads, or cross-job cache sharing. Flag one occurrence for each dependency-cache step that satisfies these conditions. Do not flag cache steps for unrelated ecosystems, build outputs, generated artifacts, compiler caches, Docker layers, tool downloads, cross-job cache sharing, or setup actions without a clear built-in dependency caching option. Do not flag cases where the setup action, cache target, or ecosystem match cannot be determined confidently. When uncertain, do not flag.

**Review notes**

- Only flag `actions/cache` when there is a corresponding setup action in the SAME workflow that already offers built-in caching (e.g., `actions/setup-node` with `cache: npm`).
- Do NOT flag `actions/cache` when the cache target is build output, generated artifacts, compiler cache (ccache, sccache), Docker layers, tool downloads, or cache shared across jobs.
- Known setup actions with built-in caching: `actions/setup-node`, `actions/setup-python`, `actions/setup-java`, `actions/setup-go`, `ruby/setup-ruby`.
- If the `actions/cache` is for a different ecosystem than the setup action present, do NOT flag.

**Examples**

- **With smell**: `actions/cache` for npm together with `setup-node`. Use `cache: npm` in the setup.

  ```yaml
  steps:
    - uses: actions/setup-node@v4
      with:
        node-version: 20
    - uses: actions/cache@v4
      with:
        path: ~/.npm
        key: npm-${{ hashFiles('package-lock.json') }}
    - run: npm ci
  ```

- **With smell**: `actions/cache` for pip together with `setup-python`. Use `cache: pip` in the setup.

  ```yaml
  steps:
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    - uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: pip-${{ hashFiles('requirements.txt') }}
    - run: pip install -r requirements.txt
  ```

- **With smell**: `actions/cache` for Maven together with `setup-java`. Use `cache: maven` in the setup.

  ```yaml
  steps:
    - uses: actions/setup-java@v4
      with:
        java-version: '17'
        distribution: temurin
    - uses: actions/cache@v4
      with:
        path: ~/.m2/repository
        key: maven-${{ hashFiles('pom.xml') }}
    - run: mvn verify
  ```

- **Without smell**: `setup-node` with built-in `cache: npm` needs no manual `actions/cache`.

  ```yaml
  steps:
    - uses: actions/setup-node@v4
      with:
        node-version: 20
        cache: npm
    - run: npm ci
  ```

- **Without smell**: `actions/cache` for build output (dist/) is not a dependency cache.

  ```yaml
  steps:
    - uses: actions/setup-node@v4
      with:
        node-version: 20
        cache: npm
    - uses: actions/cache@v4
      with:
        path: dist/
        key: build-${{ github.sha }}
  ```

- **Without smell**: `actions/cache` for Docker layers is not a language dependency cache.

  ```yaml
  steps:
    - uses: actions/cache@v4
      with:
        path: /tmp/.buildx-cache
        key: docker-${{ hashFiles('Dockerfile') }}
    - uses: docker/build-push-action@v5
  ```

- **Without smell**: `actions/cache` for a different ecosystem than the setup action. No match.

  ```yaml
  steps:
    - uses: actions/setup-node@v4
      with:
        node-version: 20
        cache: npm
    - uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: pip-${{ hashFiles('requirements.txt') }}
  ```


### 16. Use single-command steps

- **Problem:** Multiple commands per step reduce clarity and complicate debugging.
- **Solution:** Split complex steps into simpler single-command steps.

**Definition: flagging criterion**

> A `run:` step exhibits this smell only if ALL conditions are true: (1) the step contains two or more relevant shell commands in the same `run:` block; (2) at least two of these commands belong to different workflow phases, such as install, build, test, lint, package, deploy, upload, or cleanup; and (3) these phase-related commands could reasonably be executed as separate `run:` steps. Relevant commands are commands that perform meaningful workflow operations. Do not count auxiliary shell constructs or helper commands by themselves, such as `echo`, `cd`, `pwd`, `ls`, `mkdir`, `export`, variable assignments, conditionals, loops, comments, `true`, `false`, or `sleep`, unless they clearly implement a distinct workflow phase. Flag one occurrence for each `run:` step that satisfies these conditions. Do not flag single relevant commands, commands split across lines only for readability, script invocations such as `npm test` or `python script.py`, short condition/fallback commands for one operation, tightly coupled commands that perform one task, or cases where the commands, phases, or separability cannot be determined confidently.

**Review notes**

- We should not count commands such as `echo`, `if`, `cd`, `pwd`, `ls`, `mkdir`, `export`, variable assignments, loops, comments, `true`, `false`, or `sleep` as occurrences of this smell; count only the commands that are relevant or that could run in another `run` block.
- The key criterion is that the commands belong to different PHASES of the workflow (install, build, test, lint, deploy…). Coupled commands that together perform a single task do not constitute a smell.
- When in doubt about this smell, ask in the validation group.

**Examples**

- **With smell**: Three commands from different phases (install, test, build) in the same `run:` block.

  ```yaml
  steps:
    - run: |
        npm install
        npm test
        npm run build
  ```

- **With smell**: Install + deploy chained with `&&`. Different phases in the same step.

  ```yaml
  steps:
    - run: pip install -r requirements.txt && python deploy.py
  ```

- **Without smell**: Each command in its own step. Correct separation.

  ```yaml
  steps:
    - name: Install
      run: npm install
    - name: Test
      run: npm test
    - name: Build
      run: npm run build
  ```

- **Without smell**: `echo` + relevant command. An auxiliary does not count as a separate phase.

  ```yaml
  steps:
    - run: |
        echo "Starting build..."
        npm run build
  ```

- **Without smell**: Coupled commands that perform a single task (configure + install deps).

  ```yaml
  steps:
    - run: |
        cd packages/app
        npm ci
  ```

- **Without smell**: A single script invocation, not multiple phase commands.

  ```yaml
  steps:
    - run: python scripts/run_tests.py
  ```


### 17. Run tests on multiple OS's

- **Problem:** Testing on single OS might miss OS-specific issues.
- **Solution:** Use matrix strategy to run on multiple OSs.
- **How to identify:** Test jobs should run on multiple operating systems via `strategy.matrix`. If you see only `runs-on: ubuntu-latest` in a test job, it is testing on a single OS.

**Definition: flagging criterion**

> A job exhibits this smell only if ALL conditions are true: (1) the job is a test job, identified by test-related job names or commands such as `test`, `pytest`, `npm test`, `yarn test`, `mvn test`, `gradle test`, `go test`, `cargo test`, or equivalent; (2) the job has exactly one effective operating system in its `runs-on` configuration, or equivalent runner configuration; and (3) the job does not define an OS matrix or equivalent strategy with two or more distinct operating system values. Flag one occurrence for each test job that satisfies these conditions. If the job defines a matrix, but the matrix contains only one distinct operating system value, still flag the job because it effectively tests on only one OS. Do not flag non-test jobs, deployment or documentation jobs, jobs that effectively test on two or more operating systems, jobs running inside a fixed container environment, or cases where the job purpose or operating system configuration cannot be determined confidently. When uncertain, do not flag.

**Caution / Gotcha**

> Make sure the workflow or job is a TEST one.


### 18. Specify package versions

- **Problem:** Unspecified versions can lead to non-reproducible builds.
- **Solution:** Specify exact package versions in install commands.

**Definition: flagging criterion**

> A `run:` step exhibits this smell only if ALL conditions are true: (1) it installs external packages, tools, or dependencies using commands such as `apt install`, `apt-get install`, `brew install`, `pip install`, `pip install -r`, `npm install -g`, `gem install`, `cargo install`, or similar; (2) at least one installed package/tool is not explicitly pinned to a fixed version, commit SHA, digest, checksum-verified artifact, immutable release URL, or fixed version constraint; and (3) the installation can affect build, test, analysis, deployment, or release behavior. Treat dependency files such as `requirements.txt` as unpinned unless their contents are available and every dependency is pinned to an exact version. Do not flag package index updates without installation, installs from pinned commits/digests/immutable releases, installs with exact version constraints, or cases governed by verified lockfiles such as `package-lock.json`, `yarn.lock`, `poetry.lock`, `Pipfile.lock`, `Gemfile.lock`, or equivalent. When a referenced dependency file cannot be inspected, flag it as unpinned. Flag one occurrence for each `run:` step that satisfies all conditions.

**Review notes**

- We should count as a smell and occurrence cases that use `requirements.txt` or equivalent (versions may not be pinned).
- We should NOT count as a smell `run` cases that use some kind of lockfile (`package-lock.json`, `yarn.lock`, `poetry.lock`, `Pipfile.lock`, `Gemfile.lock`, etc.).
- When the referenced dependency file cannot be inspected, treat it as unpinned.

**Examples**

- **With smell**: Packages installed without a fixed version, so they pull the latest available version.

  ```yaml
  steps:
    - run: pip install pytest
    - run: npm install -g eslint
  ```

- **With smell**: `pip install -r requirements.txt`. The file contents cannot be verified, treat as unpinned.

  ```yaml
  steps:
    - run: pip install -r requirements.txt
  ```

- **With smell**: `apt-get install` without a fixed version for a tool that affects the build.

  ```yaml
  steps:
    - run: sudo apt-get install -y jq curl
  ```

- **Without smell**: Versions explicitly pinned for each package.

  ```yaml
  steps:
    - run: pip install pytest==8.2.0
    - run: npm install -g eslint@9.7.0
  ```

- **Without smell**: `npm ci` uses `package-lock.json`, governed by a lockfile.

  ```yaml
  steps:
    - run: npm ci
  ```

- **Without smell**: `yarn install --frozen-lockfile`, a verified lockfile.

  ```yaml
  steps:
    - run: yarn install --frozen-lockfile
  ```

- **Without smell**: Index update without installation does not constitute a smell.

  ```yaml
  steps:
    - run: sudo apt-get update
  ```


### 19. Run CI on multiple language versions

- **Problem:** Single version might miss version-specific issues.
- **Solution:** Use matrix strategy for multiple language versions.
- **How to identify:** `actions/setup-node|python|java|dotnet` steps with a single version (`node-version: 20`) test only one language version. Ideally use `strategy.matrix` with multiple versions.

**Definition: flagging criterion**

> A CI job exhibits this smell only if ALL conditions are true: (1) the job performs CI validation, such as testing, building, linting, static analysis, or integration checks; (2) the job configures exactly one explicit project language or runtime version in the workflow file; and (3) the job does not define a matrix or equivalent strategy that runs the same CI validation on two or more distinct project language/runtime versions. This smell concerns only project language or runtime versions, not operating systems. Do not treat `runs-on`, OS labels such as `ubuntu-latest`, `windows-latest`, or `macos-latest`, or OS matrices as runtime-version coverage. Treat versions configured through project runtime setup actions, such as `actions/setup-node` with `node-version`, `actions/setup-python` with `python-version`, `actions/setup-java` with `java-version`, `actions/setup-go` with `go-version`, `ruby/setup-ruby` with `ruby-version`, or equivalent setup actions, as explicit runtime versions. If a matrix exists but contains only one distinct runtime version, still flag the job. Do not flag non-CI jobs, deployment-only jobs, documentation jobs, jobs already testing two or more distinct runtime versions, jobs without an explicit project runtime version in the workflow file, jobs that configure only auxiliary tool versions, or jobs whose CI purpose or runtime-version configuration cannot be determined confidently. When uncertain, do not flag. Flag one occurrence for each CI job that satisfies all conditions.
