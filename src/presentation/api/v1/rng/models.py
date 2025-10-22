from typing import Literal, Annotated
from fastapi import Query

from pydantic import BaseModel, Field


class GenerateRequestSchema(BaseModel):
    from_num: str = Field(0, alias="from")
    to_num: str = Field(0, alias="to")
    count: int = 100
    base: int = 2
    format: Literal['json', 'txt'] = 'json'

GEN_REQ_SCHEMA = Annotated[GenerateRequestSchema, Query()]


class GeneratorResponseSchema(BaseModel):
    ...
