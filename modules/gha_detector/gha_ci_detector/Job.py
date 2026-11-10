
from gha_ci_detector.Step import Step


class Job:
    def __init__(self, name, yaml):
        self.name = name
        self.yaml: dict = yaml

    @property
    def job_name(self):
        if "name" in self.yaml:
            return self.yaml["name"]
        else:
            return self.name

    def has_permissions(self):
        return "permissions" in self.yaml

    def get_steps(self) -> list[Step]:
        if "steps" in self.yaml:
            return [Step(x) for x in self.yaml["steps"]]
        else:
            return [Step({
                "uses": self.yaml["uses"]
            }, True)]

    def get_if(self) -> str | None:
        if "if" in self.yaml:
            value = self.yaml["if"]
            return str(value) if not isinstance(value, str) else value
        else:
            return None

    def __str__(self):
        return "Job: \n" + self.name + " " + str(self.yaml) + "\n"

    def __repr__(self):
        return self.__str__()
