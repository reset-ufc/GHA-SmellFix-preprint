import re

from yamllint import config, linter

from gha_ci_detector.Job import Job
from gha_ci_detector.Step import Step
from gha_ci_detector.Workflow import Workflow


def files_should_be_indented_correctly(workflow: Workflow) -> None:
    rules = """
extends: default

rules:
    document-start: disable
    line-length:
        max: 400
    truthy:
        check-keys: false

    """
    yaml_config = config.YamlLintConfig(content=rules)
    problems = list(linter.run(workflow.file_content, yaml_config))
    if len(problems) > 0:
        workflow.smells.add("14. Avoid incorrectly formatted workflows")
    workflow.styling = problems
    # Maybe this needs to be renamed to correctly formatted workflows?


def external_actions_must_have_permissions_workflow(workflow: Workflow) -> None:
    """
    Check if the change adds 'permission' and if we are using an external action.
    Make sure we distinguish between job and workflow
    :param workflow:
    :return:
    """
    if "permissions" in workflow.yaml:
        return

    # Are we using secrets?
    jobs_using_secrets = []
    if ("secrets.GITHUB_TOKEN" in str(workflow.yaml) or
            "secrets.GH_BOT_ACCESS_TOKEN" in str(workflow.yaml)):
        jobs_using_secrets = list(filter(lambda j: "secrets.GITHUB_TOKEN" in str(j.yaml)
                                                   or "secrets.GH_BOT_ACCESS_TOKEN" in str(j.yaml),
                                         workflow.get_jobs()))
    # We have jobs with secrets.GITHUB_TOKEN
    if len(jobs_using_secrets) > 0:
        # There is no global permissions set
        for job in jobs_using_secrets:
            if not job.has_permissions():
                line_nr = workflow.get_line_number(job.name + ":", use_whitespace=False)
                workflow.smells.add("15. Use permissions whenever using Github Token (job at line "
                                    f"{line_nr})")

    for job in workflow.get_jobs():
        if job.has_permissions():
            continue
        for step in job.get_steps():
            if "uses" in step.yaml:
                line_nr = workflow.get_line_number(job.name + ":", use_whitespace=False)
                workflow.smells.add("6. Define permissions for workflows with external actions ("
                                    "job "
                                    f"at line: {line_nr})")


def pull_based_actions_on_fork(workflow: Workflow) -> None:
    """
    Check if the 'if' statement is added somewhere, also make sure that the action we are doing
    is 'the correct one'
    TODO: There is a paper classifying workflows, check what they did?
    :param workflow:
    :return:
    """

    def is_pull_based_name(name) -> bool:
        if name is None:
            return False
        return (" pr " in name or "_pr" in name or "pr_" in name or "issue" in name or
                "review" in name or "branch" in name or "pull request" in name
                or "pull_request" in name or "pull-request" in name or "label" in name)

    pull_based_workflow = ("name" in workflow.yaml
                           and is_pull_based_name(workflow.yaml["name"])) or is_pull_based_name(
        workflow.name)

    for job in workflow.get_jobs():
        job_has_if = job.get_if() is not None and ("github.repository" in job.get_if() or
                                                   "github.repository_owner" in job.get_if() or
                                                   "repo.full_name" in job.get_if())
        if is_pull_based_name(job.name) or pull_based_workflow:
            line_nr = workflow.get_line_number(f"{job.name.strip()}:".replace(" ", ""),
                                               use_whitespace=False)
            # We expect an if statement on the job
            if not job_has_if:
                workflow.smells.add(f"2. Prevent running issue/PR actions on forks (job line"
                                    f": {line_nr})")
        # Otherwise we need to check
        elif not job_has_if:
            for step in job.get_steps():
                step_has_if = step.get_if() is not None and ("github.repository" in step.get_if() or
                                                             "github.repository_owner" in
                                                             step.get_if() or "repo.full_name"
                                                             in step.get_if())
                if ((is_pull_based_name(str(step.get_execution())) or is_pull_based_name(str(
                        step.get_name()))) and not step_has_if):
                    (start, end) = step.get_line_numbers(workflow.get_line_number)
                    workflow.smells.add(f"2. Prevent running issue/PR actions on forks line "
                                        f"{start}:{end}")

            # TODO: Maybe we can extend this further?


def running_ci_when_nothing_changed(workflow: Workflow) -> None:
    """
    CI includes building, testing, linting.
    TODO: Double check that we are actually doing some CI
    related tasks
    :param workflow:
    :return:
    """

    def contains_path(on: dict) -> bool:
        return isinstance(on, dict) and ("paths" in on or "paths-ignore" in on)


    def independent_on_code_change(on: dict) -> bool:
        return "push" in on

    ci_list = ["lint", "build", "test", "compile", "style", "ci", "codeql", "cypress"]
    is_ci_file_name = any(word in workflow.name.lower() for word in ci_list)
    is_ci_in_workflow = any(
        word in str(workflow.yaml).lower() for word in ci_list)

    if is_ci_file_name or is_ci_in_workflow:
        if ("on" in workflow.get_keys() and isinstance(workflow.yaml["on"], dict)
                and independent_on_code_change(workflow.yaml["on"])):
            if not contains_path(workflow.get_on()["push"]):
                workflow.smells.add("16. Avoid running CI related actions when no source code has "
                                    "changed")
        elif "on" not in workflow.get_keys():
            workflow.smells.add("16. Avoid running CI related actions when no source code has "
                                "changed")


def use_fixed_version_runs_on(workflow: Workflow) -> None:
    """
    Runs on should use a fixed version and not 'latest'
    :param workflow:
    :return:
    """
    lines = workflow.file_content.split("\n")
    runs_on_lines = list(filter(lambda x: "runs-on" in x and "latest" in x, lines))
    for line in runs_on_lines:
        line_nr = lines.index(line)
        workflow.smells.add(f"3. Use fixed version for runs-on argument (line {line_nr})")


def use_specific_version_instead_of_dynamic(workflow: Workflow) -> None:
    """
    Check if a version is updated to contain more dots or is changed from latest to something
    else or is updated to be a hash value
    Using tags is as versions for actions is a known security problem
    :param workflow:
    :return:
    """
    lines = workflow.file_content.split("\n")
    uses_lines = filter(lambda x: "uses:" in x, lines)
    for line in uses_lines:
        line_nr = lines.index(line)
        if "@" not in line:
            continue
        versions = [x.strip() for x in line.split("#", 1)[0].split("@")]
        # Make sure that a 'uses' is not commented
        if versions == ['']:
            continue
        if len(versions) == 1:
            workflow.smells.add(f"8. Use commit hash instead of tags for action versions (line "
                                f"{line_nr})")
            continue
        if len(versions) >= 2 and ("v" in versions[1] or "." in versions[1]):
            workflow.smells.add(f"8. Use commit hash instead of tags for action versions (line "
                                f"{line_nr})")


def action_should_have_timeout(workflow: Workflow) -> None:
    """
    TODO: Try to compile a list of actions on github which tend to run long?
          Or try to compile a list of actions which access the outside world?
          Differentiate between jobs having a timeout and steps having a timeout.
          Jobs should be good practice and specific steps should be smell?
    :param change:
    :return:
    """
    for job in workflow.get_jobs():
        # We are purely running a different workflow so that should have this config
        if "steps" not in job.yaml:
            continue
        if "timeout-minutes" not in job.yaml:
            line_nr = workflow.get_line_number(job.name + ":", use_whitespace=False)
            workflow.smells.add(f"10. Avoid jobs without timeouts (line: {line_nr})")


def use_cache_from_setup(workflow: Workflow) -> None:
    """
    Many setup/install actions such as `setup-node` already provide a cache for the downloaded libraries
    Should it be desirable to have the cache param even when they are not yet doing caching?
    :param workflow:
    :return:
    """
    cacheable_actions = ["actions/setup-python", "actions/setup-java", "actions/setup-node"]
    for job in workflow.get_jobs():
        is_cachable_action = False
        is_cache_action = False
        for _index, step in enumerate(job.get_steps()):
            is_cachable_action = (("uses" in step.yaml and any(action in str(step.yaml) for
                                                                     action in cacheable_actions)
                                  and "cache" not in str(step.yaml))
                                  or is_cachable_action)
            is_cache_action = (("uses" in step.yaml and "actions/cache" in step.get_uses()
                                and any(keyword in str(step.yaml) for keyword in ["pip",
                                                                                  "python",
                                                                                  "requirements.txt", "maven", "pom.xml", "gradle", "build.gradle", "npm", "package-lock", "yarn"]))
                               or is_cache_action)

        if is_cache_action and is_cachable_action:
            workflow.smells.add("21. Use cache parameter instead of cache option")


def scheduled_workflows_on_forks(workflow: Workflow) -> None:
    on_dict = workflow.get_on()
    if on_dict is not None and isinstance(on_dict, dict) and "schedule" in on_dict:
        # We are dealing with a cron workflow
        for job in workflow.get_jobs():
            if_statements = ["github.repository", "github.repository_owner", "repo.full_name "]
            if job.get_if() is None:
                workflow.smells.add("1. Avoid executing scheduled workflows on forks")
                continue
            if not any(word in job.get_if() for word in if_statements):
                workflow.smells.add("1. Avoid executing scheduled workflows on forks")


def use_name_for_step(workflow: Workflow) -> None:
    for job in workflow.get_jobs():
        for step in job.get_steps():
            if "name" not in step.yaml and not step.is_inherited:
                (start, end) = step.get_line_numbers(workflow.get_line_number)
                workflow.smells.add(f"13. Use names for run steps (lines {start}:{end})")


def upload_artifact_must_have_if(workflow: Workflow) -> None:
    for job in workflow.get_jobs():
        if job.get_if() is not None and "github.repository" in job.get_if():
            continue
        for step in job.get_steps():
            if "uses" in step.yaml and ("actions/upload-artifact" in step.yaml["uses"] or
                                               "coverallsapp/github-action" in step.yaml["uses"] or
                                               "codecov/codecov-action" in step.yaml["uses"]):
                if step.get_if() is None:
                    stripped = step.yaml["uses"].strip()
                    line_nr = workflow.get_line_number(f"uses: {stripped}".replace(" ", ""),
                                                       use_whitespace=False)
                    workflow.smells.add(f"7. Use 'if' for upload-artifact action (line {line_nr})")
                else:
                    if not (("github.repository" in step.get_if() or "github.repository_owner"
                             in step.get_if()) or (
                                    job.get_if() is not None and not (
                                    "github.repository" in job.get_if()
                                    or "github.repository_owner" in job.get_if()))):
                        stripped = step.yaml["uses"].strip()
                        line_nr = workflow.get_line_number(f"uses: {stripped}".replace(" ", ""),
                                                       use_whitespace=False)
                        workflow.smells.add(f"11. Avoid uploading artifacts on forks (line"
                                            f" {line_nr})")
            elif step.get_name() is not None and "upload" in step.get_name().lower():
                if (step.get_if() is not None and not ("github.repository" in step.get_if() or
                                                       "github.repository_owner" in step.get_if())):
                    # if ((step.get_if() is None)
                    #         or not (("github.repository" in step.get_if() or
                    #                                     "github.repository_owner"
                    #                                     in step.get_if()))):
                    (start, end) = step.get_line_numbers(workflow.get_line_number)
                    workflow.smells.add(
                        f"11. Avoid uploading artifacts on forks (line {start}:{end}) for job"
                        f" {job.name}")


def multi_line_steps(workflow: Workflow) -> None:
    """
    TODO: This smell still needs to be renamed
    :param workflow:
    :return:
    """
    for job in workflow.get_jobs():
        for step in job.get_steps():
            if "run" in step.yaml:
                run = step.yaml["run"]
                unformatted_run = run.replace("\\\n", " ")
                if "\n" in unformatted_run[:-1] or "&&" in run:
                    line_nr = workflow.get_line_number(("-run: " + run.split("\n")[0]).strip(),
                                                       use_whitespace=False)
                    workflow.smells.add(f"9. Steps should only perform a single command (line "
                                        f"{line_nr})")


def comment_in_workflow(workflow: Workflow) -> None:
    source_code = workflow.file_content
    if "#" not in source_code:
        workflow.smells.add("12. Avoid workflows without comments")


def deploy_from_fork(workflow: Workflow) -> None:
    if "deploy" in workflow.name:
        for job in workflow.get_jobs():
            if job.get_if() is None:
                workflow.smells.add("22. Avoid deploying jobs on forks")
                continue
            if ("github.repository" not in job.get_if() and
                                        "github.repository_owner" not in job.get_if()):
                workflow.smells.add("22. Avoid deploying from forks")

    for job in workflow.get_jobs():
        if job.get_if() is None:
            workflow.smells.add("22. Avoid deploying jobs on forks")
            continue
        if ("github.repository" not in job.get_if() and
                "github.repository_owner" not in job.get_if()):
            workflow.smells.add("22. Avoid deploying jobs on forks")


def run_multiple_versions(workflow: Workflow) -> None:
    def job_has_setup_action_with_version(job: Job) -> bool:
        setup_step: list[Step] = list(filter(lambda s: s.get_uses() is not None and
                                                      "actions/setup" in
                                                  s.get_uses(), job.get_steps()))
        if len(setup_step) == 0:
            return True
        for step in setup_step:
            if "node" in step.get_uses():
                if "with" in step.yaml and "node-version" in step.yaml["with"]:
                    return "matrix" in str(step.yaml["with"]["node-version"])
                else:
                    return False
            elif "java" in step.get_uses():
                if "with" in step.yaml and "java-version" in step.yaml["with"]:
                    return "matrix" in str(step.yaml["with"]["java-version"])
            elif "dotnet" in step.get_uses():
                if "with" in step.yaml and "dotnet" in step.yaml["with"]:
                    return "matrix" in str(step.yaml["with"]["java-version"])
                else:
                    return False
            else:
                return True

    has_build = "build" in workflow.name.lower() or "test" in workflow.name.lower()
    for job in workflow.get_jobs():
        if has_build or "build" in str(job.job_name).lower() or "test" in str(job.job_name).lower():
            if ("runs-on" in job.yaml and ("matrix" not in job.yaml["runs-on"]) and ","
                    not in str(job.yaml["runs-on"])):
                workflow.smells.add(f"19. Run tests on multiple OS's (job: {job.name})")
            if not job_has_setup_action_with_version(job):
                workflow.smells.add(f"20. Run CI on multiple language versions (job: {job.name})")


def installing_packages_without_version(workflow: Workflow) -> None:
    def excluded_commands(line: str) -> bool:
        return "upgrade" not in line and "mvn" not in line and ".sh" not in line

    def included_commands(line: str) -> bool:
        return ("npm" in line or "npx" in line or "pip" in line or "brew" in
                line)

    for job in workflow.get_jobs():
        for step in job.get_steps():
            if "run" in step.yaml:
                run = step.yaml["run"]
                lines = run.split("\n")
                for l in lines:
                    # Special case for playwright because it cannot handle versions
                    if "npx playwright install" in l:
                        continue
                    if (" install " in l and len(l.split(" ")) >= 3 and excluded_commands(l) and
                            included_commands(l)):
                        version = re.search("(((=|@)[0-9]+(.[0-9]+.[0-9]+)?))", l)
                        if version is None and "--channel" not in l and "latest" not in l:
                            line_nr = workflow.get_line_number("-run:" + l.strip(),
                                                               use_whitespace=False)
                            workflow.smells.add("18. Avoid installing packages without version ("
                                                "line "
                                                f"{line_nr})")


def stop_workflows_for_old_commit(workflow: Workflow) -> None:
    if "concurrency" not in workflow.yaml:
        if ((isinstance(workflow.yaml["on"], dict) and "schedule" in workflow.get_on()) or
                "release" in workflow.name.lower()):
            workflow.smells.add(
                "17. Avoid starting new workflow whilst the previous one is still running")
        if (workflow.get_on() is not None
                and isinstance(workflow.get_on(), dict) and "push" in workflow.get_on()):
            workflow.smells.add("4. Stop running workflows when there is a newer commit in branch")
        if workflow.get_on() is not None and isinstance(workflow.get_on(), dict) and (
                "pull_request" in workflow.get_on() or "pull_request_target" in
                workflow.get_on()):
            workflow.smells.add("5. Stop running workflows when there is a newer commit in PR")
