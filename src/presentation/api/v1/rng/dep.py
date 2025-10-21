from typing import Annotated

from fastapi import Depends

from src.core.config import env_config, app_config
from src.integrations.locusonus.api import LocusonusAPI
from src.integrations.locusonus.cache import LocusonusCacheKeeper
from src.integrations.locusonus.client import LocusonusClient
from src.integrations.redis.client import RedisClient
from src.services.rng.rng import RNG
from src.services.rng.service import RngService

async def get_locusonus_cache_keeper() -> LocusonusCacheKeeper:
    return LocusonusCacheKeeper(
                redis_client=RedisClient(
                    env_config.REDIS_URL,
                ),
                ttl=app_config.LOCUSONUS_API_TTL,
            )

async def get_locusonus_api() -> LocusonusAPI:
    return LocusonusAPI(
                url=app_config.LOCUSONUS_API_URL,
                timeout=app_config.LOCUSONUS_API_TIMEOUT,
            )

async def get_locusonus_client(
        client: LocusonusAPI = Depends(get_locusonus_api),
        cache_keeper: LocusonusCacheKeeper = Depends(get_locusonus_cache_keeper),
) -> LocusonusClient:
    return LocusonusClient(
            client=client,
            cache_keeper=cache_keeper,
        )

async def get_rng(source_getter: LocusonusClient = Depends(get_locusonus_client)) -> RNG:
    return RNG(
        source_getter=source_getter,
    )

async def get_rng_service(rng: RNG = Depends(get_rng)) -> RngService:
    return RngService(
        rng=rng
        )


RNG_SERVICE_DEP = Annotated[RngService, Depends(get_rng_service)]