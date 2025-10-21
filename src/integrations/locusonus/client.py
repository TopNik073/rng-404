from http.client import HTTPException

from src.integrations.locusonus.api import LocusonusAPI
from src.integrations.locusonus.cache import LocusonusCacheKeeper
from src.integrations.locusonus.models import LocusonusResponseModel


class LocusonusClient:
    def __init__(self, client: LocusonusAPI, cache_keeper: LocusonusCacheKeeper):
        self.client = client
        self.cache_keeper = cache_keeper

    async def get_sources(self) -> list[LocusonusResponseModel]:
        sources: list[LocusonusResponseModel] | None = await self.cache_keeper.get_sources()
        if not sources:
            sources: list[LocusonusResponseModel] = await self.client.get_sources()
            if not sources:
                raise HTTPException("Can't get sources from Locusonus")

            await self.cache_keeper.set_sources(sources)

        return sources
