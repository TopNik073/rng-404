from fastapi import APIRouter, UploadFile, File

from src.presentation.api.v1.rng.dep import RNG_SERVICE_DEP
from src.presentation.api.v1.rng.models import GEN_REQ_SCHEMA

rng_router = APIRouter(prefix='/rng')

@rng_router.get('/generate')
async def rng_generator(
        rng_service: RNG_SERVICE_DEP,
        params: GEN_REQ_SCHEMA,
):
    return await rng_service.generate()


@rng_router.post('/generate/upload-audio')
async def rng_generator_upload_audio(
        rng_service: RNG_SERVICE_DEP,
        file: UploadFile = File(...),  # noqa
):
    return await rng_service.upload_audio(file)
