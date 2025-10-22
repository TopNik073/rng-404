from typing import Literal, Annotated
from fastapi import Query

from pydantic import BaseModel


class GenerateRequestSchema(BaseModel):
    from_num: str = Query(..., alias="from")
    to_num: str = Query(..., alias="to")
    count: int = 5
    base: int = 10
    uniq_only: bool = True
    format: Literal['json', 'txt'] = 'json'

GEN_REQ_SCHEMA = Annotated[GenerateRequestSchema, Query()]


class GeneratorResponseSchema(BaseModel):
    ...
