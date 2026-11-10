"""UserTurnBuilder — typed XML input slots per role (#46).

The single source for the user-turn data-demarcation contract. Each role
declares its ordered, typed :class:`Slot`s; :meth:`UserTurnBuilder.build` wraps
the runtime values in the matching XML tags (via ``utils.xml_wrap``). This
retires the scattered inline ``xml_wrap(...)`` builders that lived in
``prompts.py`` and the hand-maintained dynamic-placeholder denylist —
``prompts`` now derives that denylist from :func:`all_user_turn_tags`.

GitHub Actions workflow input validation (size / YAML / required-key) lives here
too, co-located with the slot contract: validating the inbound workflow and
declaring how it is demarcated for the model are one concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from .errors import WorkflowValidationError
from .utils import xml_wrap


@dataclass(frozen=True, slots=True)
class Slot:
    """One typed user-turn input: an XML tag plus how an absent value renders.

    A required slot must be supplied at build time. An optional slot renders its
    ``default`` ("None") when the value is absent or falsy — matching the legacy
    ``value or "None"`` behaviour the retry-feedback slots relied on. An optional
    slot with ``omit_if_absent=True`` instead renders *no block at all* when its
    value is absent or falsy, for always-present-or-nothing metadata (e.g.
    ``repository_context``) where a literal "None" would be noise rather than the
    meaningful "no feedback yet" signal the retry slots carry.
    """

    tag: str
    required: bool = True
    default: str = "None"
    omit_if_absent: bool = False


class UserTurnBuilder:
    """Builds one role's user turn from its ordered :class:`Slot`s."""

    def __init__(self, *slots: Slot) -> None:
        self.slots = slots

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(slot.tag for slot in self.slots)

    def build(self, **values: str) -> str:
        """Wrap ``values`` in their slots' XML tags, in declared order.

        Required slots must be present (and non-``None``); optional slots fall
        back to their ``default`` when absent or falsy. Unknown keys are
        rejected so a caller can never smuggle an undocumented tag into the turn.
        """
        unknown = set(values) - set(self.tags)
        if unknown:
            raise ValueError(
                f"unknown user-turn slot(s) {sorted(unknown)}; expected {list(self.tags)}"
            )
        blocks: list[str] = []
        for slot in self.slots:
            if slot.required:
                if values.get(slot.tag) is None:
                    raise ValueError(f"missing required user-turn slot {slot.tag!r}")
                value = values[slot.tag]
            else:
                raw = values.get(slot.tag)
                if not raw and slot.omit_if_absent:
                    continue
                value = raw or slot.default
            blocks.append(xml_wrap(slot.tag, value))
        return "\n\n".join(block for block in blocks if block.strip())


# The workflow's repository metadata, carried alongside the workflow YAML it
# describes. Omitted entirely (not rendered as "None") when an ground_truth entry lacks
# it, so an unenriched ground_truth degrades gracefully — see :class:`Slot`. Every role
# that receives the workflow also receives this block, immediately after it.
_REPO_CONTEXT_SLOT = Slot("repository_context", required=False, omit_if_absent=True)

WORKFLOW_BUILDER = UserTurnBuilder(Slot("workflow"), _REPO_CONTEXT_SLOT)
DETECTION_VALIDATION_BUILDER = UserTurnBuilder(
    Slot("workflow"), _REPO_CONTEXT_SLOT, Slot("peer_detection_result")
)
D2_REFACTOR_BUILDER = UserTurnBuilder(
    Slot("original_workflow"), _REPO_CONTEXT_SLOT, Slot("detection_result")
)
REFACTOR_VALIDATION_BUILDER = UserTurnBuilder(
    Slot("original_workflow"),
    _REPO_CONTEXT_SLOT,
    Slot("detection_result"),
    Slot("proposed_refactor"),
)


_REPO_CONTEXT_FIELDS = (
    "current_repository",
    "repository_owner",
    "default_branch",
    "workflow_file_path",
)


def format_repository_context(
    *,
    current_repository: str | None = None,
    repository_owner: str | None = None,
    default_branch: str | None = None,
    workflow_file_path: str | None = None,
) -> str | None:
    """Render the ``<repository_context>`` block body from the four repo fields.

    Returns a bulleted block listing whichever fields are present, in a fixed
    order, or ``None`` when none are — so the slot is omitted rather than carrying
    a half-empty block. The four rendered fields (current_repository /
    repository_owner / default_branch / workflow_file_path) are derived by
    :func:`repository_context_from_entry` from the ground_truth's primitive columns.
    """
    values = {
        "current_repository": current_repository,
        "repository_owner": repository_owner,
        "default_branch": default_branch,
        "workflow_file_path": workflow_file_path,
    }
    lines = [f"- {name}: {values[name]}" for name in _REPO_CONTEXT_FIELDS if values[name]]
    return "\n".join(lines) if lines else None


def repository_context_from_entry(entry: dict) -> str | None:
    """Render the ``<repository_context>`` block for one ground_truth entry, or None.

    The four context fields are derived from the ground_truth's primitive columns:
    ``current_repository`` ← ``repository``, ``repository_owner`` ← the owner
    half of the ``owner/repo`` slug, ``default_branch`` direct, and
    ``workflow_file_path`` ← ``file_path``. (These were redundant pre-computed
    aliases in the legacy schema — exact 1:1 derivations — so the rendered block
    is byte-identical.) The single extraction path shared by the pipeline
    (:mod:`gha_agents.main`) and the prompt-doc dump.
    """
    current_repository = entry.get("repository")
    repository_owner = None
    if isinstance(current_repository, str) and "/" in current_repository:
        repository_owner = current_repository.split("/", 1)[0]
    return format_repository_context(
        current_repository=current_repository,
        repository_owner=repository_owner,
        default_branch=entry.get("default_branch"),
        workflow_file_path=entry.get("file_path"),
    )


USER_TURN_BUILDERS: dict[str, UserTurnBuilder] = {
    "workflow": WORKFLOW_BUILDER,
    "detection_validation": DETECTION_VALIDATION_BUILDER,
    "d2_refactor": D2_REFACTOR_BUILDER,
    "refactor_validation": REFACTOR_VALIDATION_BUILDER,
}


def all_user_turn_tags() -> frozenset[str]:
    """Every XML tag any role accepts — the single source the denylist derives from."""
    return frozenset(tag for builder in USER_TURN_BUILDERS.values() for tag in builder.tags)


_MAX_YAML_SIZE = 2 * 1024 * 1024  # 2 MB
_GITHUB_ACTIONS_KEYS = ("on", "name", "jobs", "runs-on")


def validate_workflow_input(file_key: str, content: str) -> None:
    """Validate that a YAML string is a plausible GitHub Actions workflow.

    Raises :class:`WorkflowValidationError` when the content is oversize, not
    valid YAML, not a mapping, or carries none of the top-level GitHub Actions
    keys.
    """
    if len(content.encode("utf-8")) > _MAX_YAML_SIZE:
        raise WorkflowValidationError(file_key)
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise WorkflowValidationError(file_key) from exc
    if not isinstance(parsed, dict):
        raise WorkflowValidationError(file_key)
    if not any(key in parsed for key in _GITHUB_ACTIONS_KEYS):
        raise WorkflowValidationError(file_key)
