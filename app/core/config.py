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
    # R2 설정이 없을 때 쓰는 접두어. AI 서버가 직접 서빙합니다.
    avatar_url_prefix: str = "/static/avatars"

    # Cloudflare R2. 다섯 개가 모두 있어야 업로드합니다.
    # 하나라도 비면 위 avatar_url_prefix 로 서빙합니다(로컬 개발용).
    #
    # 액세스 키는 저장소에 넣지 않습니다. 운영 서버의 .env 에만 둡니다.
    # 의류 파트가 같은 버킷의 garments/v1/ 을 쓰고 있어 접두어로 분리합니다.
    r2_endpoint: str = ""
    r2_bucket: str = ""
    r2_public_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_key_prefix: str = "avatars/v1"

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
