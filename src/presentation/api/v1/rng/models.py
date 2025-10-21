from typing import Literal, Annotated
from fastapi import Query

from pydantic import BaseModel

class GenerateRequestSchema(BaseModel):
    from_num: str = '0'
    to_num: str = '1'
    count: int = 100
    base: int = 2
    format: Literal['json', 'txt'] = 'json'

GEN_REQ_SCHEMA = Annotated[GenerateRequestSchema, Query()]


class GeneratorResponseSchema(BaseModel):
    ...
