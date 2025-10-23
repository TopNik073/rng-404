from fastapi import APIRouter
from src.presentation.api.v1.rng import rng_router
from src.presentation.api.v1.nist import nist_router

v1_router = APIRouter(prefix='/v1')
v1_router.include_router(rng_router)
v1_router.include_router(nist_router)

__all__ = [
    'v1_router',
]
