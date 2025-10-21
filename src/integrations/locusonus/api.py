from http.client import HTTPException
from typing import ClassVar

from src.core.logger import get_logger
from src.integrations.locusonus.models import LocusonusResponseModel
from httpx import AsyncClient

logger = get_logger(__name__)

class LocusonusAPI:
    endpoints: ClassVar[dict[str, str]] = {
        'get_sources': '/soundmap/list/active/json/name',
    }

    def __init__(
            self,
            url: str,
            timeout: int,
    ):
        self.url = url
        self.timeout = timeout
        self.client: AsyncClient = AsyncClient(timeout=timeout)

    async def get_sources(self) -> list[LocusonusResponseModel]:
        try:
                response = await self.client.get(self.url + self.endpoints['get_sources'])
                return [LocusonusResponseModel.model_validate(source) for source in response.json()]
        except Exception as e:
            logger.aexception(e)
            raise HTTPException("Can't get sources from Locusonus") from e
