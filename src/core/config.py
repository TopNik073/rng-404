from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # ----- APP ENV CONFIG -----
    APP_NAME: str = 'RNG-404'
    APP_HOST: str
    APP_PORT: int

    DEBUG: bool = False

    MAX_AUDIO_DURATION: float

    # ----- REDIS -----
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASS: str | None = None
    REDIS_DB: int

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASS:
            redis_url = f'redis://:{self.REDIS_PASS}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}'
        else:
            redis_url = f'redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}'

        return redis_url

    # ----- LOGGER -----
    SENSITIVE_DATA: list[str] = []
    LOG_LEVEL: str = 'INFO'
    LOG_TO_FILE: bool = False
    LOG_FORMAT: str = '%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S.%f'

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    _LOGS_DIR: Path = BASE_DIR / 'logs'

    @property
    def LOGS_DIR(self) -> Path:
        Path.mkdir(self._LOGS_DIR, parents=True, exist_ok=True)
        return self._LOGS_DIR


class AppConfig(BaseSettings):
    # ----- LOCUSONUS API -----
    LOCUSONUS_API_URL: str = 'https://locusonus.org'
    LOCUSONUS_API_TIMEOUT: float = 10.0
    LOCUSONUS_API_TTL: int = 300


env_config = EnvConfig()
app_config = AppConfig()
