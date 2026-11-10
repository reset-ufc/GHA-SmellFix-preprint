"""Table generators for the analysis notebooks."""

from utils.tables.rq1_overall_detection import overall_detection_table
from utils.tables.rq2_refactoring_effectiveness_by_design_and_model import (
    refactoring_effectiveness_by_design_and_model_table,
    refactoring_effectiveness_safety_table,
)
from utils.tables.rq1_smell_level_detection_effectiveness import (
    smell_level_detection_effectiveness_table,
)
from utils.tables.rq2_clean_removal_by_model import clean_removal_by_model_table
from utils.tables.rq2_end_to_end_conditional import end_to_end_conditional_table
from utils.tables.rq2_negative_edit_types import negative_edit_types_table
from utils.tables.rq2_refactoring_outcomes_by_smell import (
    refactoring_outcomes_by_smell_table,
)
from utils.tables.rq3_absolute_overhead import absolute_overhead_table
from utils.tables.rq3_paired_tests_across_designs import (
    paired_tests_across_designs_table,
)
from utils.tables.cost_latency_table import cost_latency_table
from utils.tables.evaluated_llms import evaluated_llms_table
from utils.tables.stats_latency_tokens import (
    stats_latency_table,
    stats_tokens_table,
)

__all__ = [
    "absolute_overhead_table",
    "clean_removal_by_model_table",
    "cost_latency_table",
    "end_to_end_conditional_table",
    "evaluated_llms_table",
    "negative_edit_types_table",
    "overall_detection_table",
    "paired_tests_across_designs_table",
    "refactoring_effectiveness_by_design_and_model_table",
    "refactoring_effectiveness_safety_table",
    "refactoring_outcomes_by_smell_table",
    "smell_level_detection_effectiveness_table",
    "stats_latency_table",
    "stats_tokens_table",
]
