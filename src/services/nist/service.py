from fastapi import UploadFile

from src.presentation.api.v1.nist.models import GenerateRequestSchema


class NistService:
    def __init__(self, nist) -> None:
        self.nist = nist

    async def check(
        self, params: GenerateRequestSchema, upload_file: UploadFile | None
    ):
        return await self.nist.run_test_suite(params, upload_file)
