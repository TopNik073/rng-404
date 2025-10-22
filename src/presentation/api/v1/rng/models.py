from typing import Literal, Annotated
from fastapi import Form, Depends

from pydantic import BaseModel

from src.integrations.locusonus.models import LocusonusResponseModel


class GenerateRequestSchema(BaseModel):
    from_num: str
    to_num: str
    count: int = 5
    base: int = 10
    uniq_only: bool = True
    format: Literal['json', 'txt'] = 'json'


async def parse_generate_params(  # noqa
    from_num: str = Form(...),
    to_num: str = Form(...),
    count: int = Form(5),
    base: int = Form(10),
    uniq_only: bool = Form(True),
    format: Literal['json', 'txt'] = Form('json'),
):
    return GenerateRequestSchema(
        from_num=from_num,
        to_num=to_num,
        count=count,
        base=base,
        uniq_only=uniq_only,
        format=format,
    )


GEN_REQ_SCHEMA = Annotated[GenerateRequestSchema, Depends(parse_generate_params)]


class GeneratorResponseSchema(BaseModel):
    executed_sources: list[LocusonusResponseModel]
    seed: str
    result: list[str]
