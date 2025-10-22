from fastapi import APIRouter, UploadFile, File

from src.presentation.api.v1.nist.dep import NIST_SERVICE_DEP
from src.presentation.api.v1.nist.models import NIST_REQ_SCHEMA

nist_router = APIRouter(prefix="/nist", tags=["Nist"])


@nist_router.post("/check")
async def nist_check_sequence(
    nist_service: NIST_SERVICE_DEP,
    params: NIST_REQ_SCHEMA,
    file: UploadFile = File(default=None),  # noqa
):
    return await nist_service.check(params=params, upload_file=file)
