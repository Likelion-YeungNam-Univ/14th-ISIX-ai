"""애플리케이션 설정.

.env 파일에서 값을 읽어 옵니다. 파일이 없으면 기본값을 사용합니다.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 서버
    port: int = 8000
    log_level: str = "INFO"

    # CORS — 쉼표로 구분해 여러 개 지정 가능
    cors_origins: str = "http://localhost:5173"

    # 모델 · 데이터 경로
    smplx_model_path: Path = Path("./assets/models/smplx/SMPLX_FEMALE.npz")
    body_grid_path: Path = Path("./assets/grid/body_grid.json")
    draped_dir: Path = Path("./assets/draped")
    measure_calibration_path: Path = Path("./assets/measure_calibration.json")

    # 처리 제한
    max_upload_size_mb: int = 10
    optimization_timeout_sec: int = 45

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
