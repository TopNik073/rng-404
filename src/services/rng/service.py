from fastapi import UploadFile

from src.presentation.api.v1.rng.models import GenerateRequestSchema


class RngService:
    def __init__(self, rng):
        self.rng = rng

    async def generate(self, params: GenerateRequestSchema, upload_file: UploadFile | None) -> str:
        return await self.rng.get_random(params, upload_file=upload_file)
