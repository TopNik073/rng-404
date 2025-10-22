from typing import Annotated, Literal
from fastapi import Query

from pydantic import BaseModel, Field

NistTestType = Literal[
    "frequency", "block_frequency", "runs", "longest_runs", "matrix_rank",
    "dft", "template", "overlapping_template", "universal", "linear_complexity",
    "serial", "approximate_entropy", "cumulative_sums", "random_excursions",
    "random_excursions_variant",
]

ALL_NIST_TESTS: list[NistTestType] = [
    "frequency", "block_frequency", "runs", "longest_runs", "matrix_rank",
    "dft", "template", "overlapping_template", "universal", "linear_complexity",
    "serial", "approximate_entropy", "cumulative_sums", "random_excursions",
    "random_excursions_variant",
]

class NistRequestSchema(BaseModel):
    sequence: str | None = Field(None)
    included_tests: list[NistTestType] = Field(default_factory=lambda: ALL_NIST_TESTS.copy())


NIST_REQ_SCHEMA = Annotated[NistRequestSchema, Query()]


class GeneratorResponseSchema(BaseModel): ...
