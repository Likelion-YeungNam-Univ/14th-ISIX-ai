"""아바타 생성 라우터.

사진 1장 + 키·몸무게로 3D 아바타와 12부위 치수를 생성합니다.

주의 — 사진 픽셀에서 직접 cm를 계산하지 않습니다.
2D 사진은 몸의 두께를 알 수 없어 둘레를 구할 수 없습니다.
사진에서는 비율만 추출하고, 3D 메시를 만든 뒤 그 위에서 측정합니다.

비동기(폴링) 구조입니다.
  POST /api/avatar/generate  -> 202 + avatar_id
  GET  /api/avatar/{id}      -> processing | done | failed

동기로 만들면 안 되는 이유가 두 가지입니다.
  1) 사진 1장 처리에 10~20초가 걸립니다. 대부분 SMPL-X beta 최적화이고,
     야코비안을 유한차분으로 잡느라 forward 를 수백 번 돌립니다. CPU 전용이라
     더 줄이기 어렵습니다.
  2) 파이프라인은 scipy·torch 로 도는 동기 CPU 작업입니다. async 함수 안에서
     그대로 호출하면 이벤트 루프를 막아 그동안 /health 를 포함한 모든 요청이
     멈춥니다. 반드시 워커 스레드/프로세스로 넘겨야 합니다.
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile, status

from app.core.config import settings
from app.core.exceptions import ClosrException, ErrorCode
from app.core.response import ApiResponse
from app.models.avatar import (
    AvatarJobResponse,
    AvatarResult,
    AvatarStatus,
    AvatarStatusResponse,
)
from app.services import avatar_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/avatar", tags=["avatar"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

# 작업 저장소.
# TODO: 인스턴스를 늘리면 프로세스 메모리로는 공유가 안 됩니다. Redis 등으로 교체할 것.
_JOBS: dict[str, AvatarStatusResponse] = {}


def _run_pipeline(avatar_id: str, photo_bytes: bytes, height: int, weight: int) -> None:
    """백그라운드 워커에서 파이프라인을 실행합니다.

    BackgroundTasks 는 동기 함수를 스레드풀에서 돌리므로 이벤트 루프를 막지
    않습니다. 이 함수를 async 로 바꾸면 안 됩니다 — 그 순간 20초 동안
    /health 를 포함한 모든 요청이 멈춥니다.
    """
    try:
        out = avatar_service.generate(photo_bytes, height, weight, avatar_id=avatar_id)
    except ClosrException as e:
        logger.warning("[%s] 아바타 생성 실패: %s", avatar_id, e.error_code.name)
        _JOBS[avatar_id] = AvatarStatusResponse(
            avatar_id=avatar_id,
            status=AvatarStatus.FAILED,
            error_message=e.error_code.message,
        )
        return
    except Exception:
        logger.exception("[%s] 아바타 생성 중 예상치 못한 오류", avatar_id)
        _JOBS[avatar_id] = AvatarStatusResponse(
            avatar_id=avatar_id,
            status=AvatarStatus.FAILED,
            error_message=ErrorCode.INTERNAL_ERROR.message,
        )
        return
    finally:
        del photo_bytes  # 원본 사진 즉시 폐기

    _JOBS[avatar_id] = AvatarStatusResponse(
        avatar_id=avatar_id,
        status=AvatarStatus.DONE,
        result=AvatarResult(
            glb_url=out.glb_url,
            body_bucket=out.body_bucket,
            measurements=out.measurements,
            confidence=out.confidence,
            warnings=out.warnings,
        ),
    )


@router.post(
    "/generate",
    response_model=ApiResponse[AvatarJobResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_avatar(
    background: BackgroundTasks,
    photo: UploadFile = File(..., description="전신 사진 (JPEG/PNG)"),
    height: int = Form(..., ge=130, le=220, description="키 (cm)"),
    weight: int = Form(..., ge=30, le=150, description="몸무게 (kg)"),
) -> ApiResponse[AvatarJobResponse]:
    """아바타 생성 작업을 등록합니다.

    결과를 기다리지 않고 avatar_id 를 즉시 반환합니다.
    완료 여부는 GET /api/avatar/{avatar_id} 로 확인하세요.

    원본 사진은 저장하지 않습니다.
    체형 파라미터 추출 후 메모리에서 즉시 폐기합니다.
    """
    if photo.content_type not in ALLOWED_CONTENT_TYPES:
        raise ClosrException(ErrorCode.UNSUPPORTED_FORMAT)

    contents = await photo.read()
    if len(contents) > settings.max_upload_bytes:
        raise ClosrException(ErrorCode.FILE_TOO_LARGE)

    avatar_id = f"av_{uuid.uuid4().hex[:12]}"
    _JOBS[avatar_id] = AvatarStatusResponse(
        avatar_id=avatar_id, status=AvatarStatus.PROCESSING
    )
    background.add_task(_run_pipeline, avatar_id, contents, height, weight)

    return ApiResponse.ok(AvatarJobResponse(avatar_id=avatar_id))


@router.get("/{avatar_id}", response_model=ApiResponse[AvatarStatusResponse])
async def get_avatar(avatar_id: str) -> ApiResponse[AvatarStatusResponse]:
    """아바타 생성 상태를 조회합니다.

    status 가 processing 이면 잠시 뒤 다시 호출하세요.
    done 이면 result 에 GLB URL·12부위 치수·신뢰도가 담깁니다.
    """
    job = _JOBS.get(avatar_id)
    if job is None:
        raise ClosrException(ErrorCode.AVATAR_NOT_FOUND)
    return ApiResponse.ok(job)
