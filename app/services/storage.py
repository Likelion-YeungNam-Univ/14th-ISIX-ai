"""생성한 GLB 를 R2 에 올리고 공개 주소를 돌려줍니다.

R2 로 옮기는 이유는 부하 분산이 아니라 HTTPS 입니다.
프론트를 Vercel 같은 곳에 배포하면 페이지가 https 인데, GLB 주소가
http://<AI서버>:8000/... 이면 브라우저가 요청 자체를 막습니다(mixed content).
3D 뷰어에 모델이 뜨지 않고 콘솔에만 조용히 경고가 남습니다.
R2 공개 주소는 https 라 이 문제가 없습니다.

설정이 없으면 업로드를 건너뛰고 None 을 돌려줍니다. 호출부는 그때
AI 서버가 직접 서빙하는 주소를 씁니다. 로컬 개발에서 R2 자격증명 없이도
돌아가야 하기 때문입니다.

업로드가 실패하면 예외를 삼키지 않고 raise 합니다. 조용히 로컬 주소로
돌아가면 운영에서 https 가 아닌 주소가 나가는데도 아무도 모릅니다.
호출부가 잡아서 경고를 응답에 실어 보냅니다.
"""
import logging
import threading
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_client = None
_lock = threading.Lock()


def is_configured() -> bool:
    """업로드에 필요한 값이 모두 있는지."""
    return all([
        settings.r2_endpoint,
        settings.r2_bucket,
        settings.r2_public_url,
        settings.r2_access_key_id,
        settings.r2_secret_access_key,
    ])


def _get_client():
    """boto3 클라이언트. R2 는 S3 호환이라 그대로 씁니다.

    클라이언트 생성이 가볍지 않아 한 번만 만들어 재사용합니다.
    업로드는 백그라운드 스레드에서 도므로 생성 구간만 잠급니다.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                import boto3
                from botocore.config import Config

                _client = boto3.client(
                    "s3",
                    endpoint_url=settings.r2_endpoint,
                    aws_access_key_id=settings.r2_access_key_id,
                    aws_secret_access_key=settings.r2_secret_access_key,
                    # R2 는 리전 개념이 없지만 boto3 가 값을 요구합니다.
                    region_name="auto",
                    config=Config(signature_version="s3v4",
                                  retries={"max_attempts": 3, "mode": "standard"}),
                )
                logger.info("R2 클라이언트 준비 완료 (bucket=%s)", settings.r2_bucket)
    return _client


def upload_glb(local_path: Path, key_name: str) -> Optional[str]:
    """GLB 를 올리고 공개 주소를 반환합니다. 설정이 없으면 None.

    key_name 은 파일명만 받습니다. 접두어는 설정에서 붙입니다.
    의류 파트가 garments/v1/... 을 쓰고 있어 아바타는 avatars/v1/... 로 둡니다.
    버전을 접두어에 둔 이유는 규칙이 바뀌었을 때 기존 파일을 지우지 않고
    v2 로 올려 롤백할 수 있게 하기 위함입니다.
    """
    if not is_configured():
        return None

    key = f"{settings.r2_key_prefix.strip('/')}/{key_name}"
    _get_client().upload_file(
        str(local_path), settings.r2_bucket, key,
        # 없으면 브라우저가 application/octet-stream 으로 받아
        # three.js 로더가 형식을 판별하지 못합니다.
        ExtraArgs={"ContentType": "model/gltf-binary"},
    )
    url = f"{settings.r2_public_url.rstrip('/')}/{key}"
    logger.info("R2 업로드 완료: %s", key)
    return url
