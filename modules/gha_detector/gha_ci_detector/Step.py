from collections.abc import Callable


class Step:
    def __init__(self, yaml, is_inherited: bool = False):
        self.yaml: dict = yaml
        self.is_inherited = is_inherited

    def get_if(self) -> str | None:
        if "if" in self.yaml:
            value = self.yaml["if"]
            return str(value) if not isinstance(value, str) else value
        else:
            return None

    def get_name(self) -> str | None:
        if "name" in self.yaml:
            return self.yaml["name"]
        else:
            return None

    def get_uses(self) -> str | None:
        if "uses" in self.yaml:
            return self.yaml["uses"]
        else:
            return None

    def get_execution(self) -> str:
        if (uses := self.get_uses()) is not None:
            return uses
        else:
            return self.yaml["run"]

    def get_line_numbers(self, get_line_number: Callable[[str, bool], int]) -> tuple[int, int]:
        numbers: list[int] = sorted(
            get_line_number(f"{k}:{self.yaml[k]}".replace(" ", ""),
                                          False) for k in self.yaml)
        return numbers[0], numbers[-1]

    def __eq__(self, other):
        if not isinstance(other, Step):
            return False
        if "uses" in self.yaml and "uses" in other.yaml:
            return self.yaml["uses"] == other.yaml["uses"]
        elif "run" in self.yaml and "run" in other.yaml:
            return self.yaml["run"] == other.yaml["run"]
        else:
            return False

