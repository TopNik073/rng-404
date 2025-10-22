from typing import Annotated

from fastapi import Depends

from src.services.nist.service import NistService


async def get_nist_service() -> NistService:
    return NistService()


NIST_SERVICE_DEP = Annotated[NistService, Depends(get_nist_service)]
