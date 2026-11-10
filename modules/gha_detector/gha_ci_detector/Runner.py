import gha_ci_detector.smell_detector as smell_detector
from gha_ci_detector.Workflow import Workflow


class Runner:
    def __init__(self, workflow: Workflow):
        self.workflow: Workflow = workflow

    def run_all(self) -> set[str]:
        smell_detector.files_should_be_indented_correctly(self.workflow)
        smell_detector.external_actions_must_have_permissions_workflow(self.workflow)
        smell_detector.pull_based_actions_on_fork(self.workflow)
        smell_detector.running_ci_when_nothing_changed(self.workflow)
        smell_detector.use_fixed_version_runs_on(self.workflow)
        smell_detector.use_specific_version_instead_of_dynamic(self.workflow)
        smell_detector.action_should_have_timeout(self.workflow)
        smell_detector.use_name_for_step(self.workflow)
        smell_detector.scheduled_workflows_on_forks(self.workflow)
        smell_detector.use_name_for_step(self.workflow)
        smell_detector.stop_workflows_for_old_commit(self.workflow)
        smell_detector.upload_artifact_must_have_if(self.workflow)
        smell_detector.multi_line_steps(self.workflow)
        smell_detector.comment_in_workflow(self.workflow)
        smell_detector.deploy_from_fork(self.workflow)
        smell_detector.run_multiple_versions(self.workflow)
        smell_detector.installing_packages_without_version(self.workflow)
        smell_detector.use_cache_from_setup(self.workflow)

        return self.workflow.smells
