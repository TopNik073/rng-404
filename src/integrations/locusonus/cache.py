import json

from src.integrations.locusonus.models import LocusonusResponseModel
from src.integrations.redis.client import RedisClient


class LocusonusCacheKeeper:
    def __init__(
            self,
            redis_client: RedisClient,
            ttl: int
    ):
        self.key: str = "locusonus_cache"
        self.redis: RedisClient = redis_client
        self.ttl = ttl

    async def get_sources(self) -> list[LocusonusResponseModel] | None:
        sources = await self.redis.get_json(self.key)
        if not sources:
            return None
        return [LocusonusResponseModel.model_validate(source) for source in json.loads(sources)]

    async def set_sources(self, sources: list[LocusonusResponseModel]) -> None:
        await self.redis.set_json(self.key, json.dumps([source.model_dump() for source in sources]), self.ttl)
