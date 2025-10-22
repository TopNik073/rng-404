from typing import Annotated
from fastapi import Query

from pydantic import BaseModel, Field


class NistRequestSchema(BaseModel):
    sequence: str = Field(None)
    included_tests: list[str] = Field(
        [
            "frequency",
            "block_frequency",
            "runs",
            "longest_runs",
            "matrix_rank",
            "dft",
            "template",
            "overlapping_template",
            "universal",
            "linear_complexity",
            "serial",
            "approximate_entropy",
            "cumulative_sums",
            "random_excursions",
            "random_excursions_variant",
        ]
    )


NIST_REQ_SCHEMA = Annotated[NistRequestSchema, Query()]


class GeneratorResponseSchema(BaseModel): ...
