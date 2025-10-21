from fastapi import APIRouter
from src.presentation.api.v1.rng import rng_router

v1_router = APIRouter()
v1_router.include_router(rng_router)

__all__ = ['v1_router',]