from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse, Response

from src.presentation.api.v1.rng.dep import RNG_SERVICE_DEP
from src.presentation.api.v1.rng.models import GEN_REQ_SCHEMA, GeneratorResponseSchema

rng_router = APIRouter(prefix='/rng', tags=['RNG'])


@rng_router.post('/generate', response_model=GeneratorResponseSchema)
async def rng_generator(
    rng_service: RNG_SERVICE_DEP,
    params: GEN_REQ_SCHEMA,
    file: UploadFile | None = File(default=None),  # noqa
) -> GeneratorResponseSchema | StreamingResponse | Response:
    return await rng_service.generate(params=params, upload_file=file)

@rng_router.get('/generate/binary-file')
async def rng_generator_binary_file(
    rng_service: RNG_SERVICE_DEP,
    length: int = Query(ge=0, default=1_000_000),
) -> StreamingResponse:
    return await rng_service.generate_binary_file(length=length)
