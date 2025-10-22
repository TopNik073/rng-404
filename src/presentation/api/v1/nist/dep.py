from typing import Annotated

from fastapi import Depends

from src.services.nist.nist import Nist
from src.services.nist.service import NistService


async def get_nist() -> NistService:
    return Nist()


async def get_nist_service(nist: Nist = Depends(get_nist)) -> NistService:
    return NistService(nist=nist)


NIST_SERVICE_DEP = Annotated[NistService, Depends(Nist)]
