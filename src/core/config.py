from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    APP_NAME: str = 'RNG-404'
    APP_HOST: str
    APP_PORT: int

    DEBUG: bool = False

    # ----- LOGGER -----
    SENSITIVE_DATA: list[str] = []
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = False
    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "[%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    _LOGS_DIR: Path = BASE_DIR / "logs"

    @property
    def LOGS_DIR(self) -> Path:
        Path.mkdir(self._LOGS_DIR, parents=True, exist_ok=True)
        return self._LOGS_DIR


env_config = EnvConfig()
