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

    # 모델 경로 — 재배포 금지 라이선스라 저장소에 없습니다. 각자 배치해야 합니다.
    smplx_model_path: Path = Path("./assets/models/smplx/SMPLX_FEMALE.npz")

    # 파트 간 계약 파일 — 저장소에 커밋됩니다.
    # 백엔드·의류 파트가 body_grid.json 을 그대로 읽어 씁니다.
    body_grid_path: Path = Path("./body_grid.json")
    bias_path: Path = Path("./bias.json")
    measure_calibration_path: Path = Path("./measure_calibration.json")

    # 산출물
    draped_dir: Path = Path("./assets/draped")
    avatar_dir: Path = Path("./assets/avatars")
    # 생성된 GLB 를 서빙할 공개 URL 접두어.
    # 운영에서는 CDN(Cloudflare R2 등) 주소로 바꾸세요.
    avatar_url_prefix: str = "/static/avatars"

    # 처리 제한
    max_upload_size_mb: int = 10
    optimization_timeout_sec: int = 45

    # 파이프라인 동시 실행 수.
    # 요청당 762MB · CPU 100% 를 10~20초 쓰므로 늘리면 서로 코어를 뺏고
    # 메모리도 부족해집니다. 2GB 인스턴스에서는 1 을 유지하세요.
    max_concurrent_pipeline: int = 1

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
