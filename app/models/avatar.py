"""아바타 응답 스키마.

부위 이름 12개는 고정입니다.
백엔드·프론트가 이 문자열을 그대로 사용하므로 임의로 바꾸지 않습니다.

아바타 생성은 비동기입니다.
POST 로 작업을 등록해 avatar_id 를 받고, GET 으로 완료 여부를 확인합니다.
사진 1장 처리에 10~20초가 걸려(대부분 SMPL-X beta 최적화) 동기 응답으로는
게이트웨이·브라우저 타임아웃에 걸리기 때문입니다.
"""

from enum import Enum

from pydantic import BaseModel, Field

MEASUREMENT_KEYS = (
    "shoulder_width",
    "chest_circ",
    "waist_circ",
    "hip_circ",
    "neck_circ",
    "arm_circ",
    "thigh_circ",
    "back_length",
    "sleeve_length",
    "inseam",
    "total_length",
    "front_width",
)

MEASUREMENT_LABELS = {
    "shoulder_width": "어깨너비",
    "chest_circ": "가슴둘레",
    "waist_circ": "허리둘레",
    "hip_circ": "엉덩이둘레",
    "neck_circ": "목둘레",
    "arm_circ": "팔둘레",
    "thigh_circ": "허벅지둘레",
    "back_length": "등길이",
    "sleeve_length": "소매길이",
    "inseam": "인심",
    "total_length": "총장",
    "front_width": "앞품",
}


class AvatarStatus(str, Enum):
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class AvatarJobResponse(BaseModel):
    """POST /api/avatar/generate 의 응답 (202 Accepted)."""

    avatar_id: str = Field(..., description="아바타 식별자. 조회에 사용합니다")
    status: AvatarStatus = Field(AvatarStatus.PROCESSING, description="작업 상태")
    poll_after_ms: int = Field(
        3000, description="이 시간 뒤부터 GET 으로 조회할 것을 권장합니다"
    )


class AvatarResult(BaseModel):
    """완료된 아바타의 내용."""

    glb_url: str = Field(..., description="3D 메시 파일 URL")
    body_bucket: str = Field(..., description="체형 12구간 코드. H{0-2}B{0-3}")
    measurements: dict[str, float] = Field(..., description="12부위 치수 (cm)")
    confidence: float = Field(..., ge=0, le=1, description="추정 신뢰도")
    warnings: list[str] = Field(default_factory=list, description="품질 경고")


class AvatarStatusResponse(BaseModel):
    """GET /api/avatar/{avatar_id} 의 응답.

    status 가 done 일 때만 result 가 채워집니다.
    failed 이면 error_message 에 사유가 담깁니다.
    """

    avatar_id: str
    status: AvatarStatus
    result: AvatarResult | None = Field(None, description="status=done 일 때만 존재")
    error_message: str | None = Field(None, description="status=failed 일 때만 존재")
