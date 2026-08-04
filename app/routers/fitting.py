"""가상 피팅 라우터.

사전 계산된 드레이핑 결과(GLB 216개)를 조회합니다.
클로스 시뮬레이션은 1벌당 30초~2분이 걸려 런타임 실행이 불가능합니다.
따라서 실시간에는 최근접 체형의 결과를 꺼내오기만 합니다.
"""

from fastapi import APIRouter, Query

from app.core.response import ApiResponse
from app.models.fitting import FittingResponse

router = APIRouter(prefix="/api/fitting", tags=["fitting"])


@router.get("/{avatar_id}", response_model=ApiResponse[FittingResponse])
async def get_fitting(
    avatar_id: str,
    garment_id: str = Query(..., description="의류 ID (예: shirt)"),
    size: str = Query("M", pattern="^[SML]$", description="사이즈"),
) -> ApiResponse[FittingResponse]:
    """아바타에 의류를 착용한 결과를 조회합니다.

    파일명 규칙 — 백엔드가 이 문자열로 파일을 조회합니다.
        착용 메시   {garment}_{size}__{bucket}.glb   shirt_M__H1B2.glb
        여유량      {garment}_{size}__{bucket}.json  shirt_M__H1B2.json

    여유량은 GLB 와 같은 배치에서 생성되므로 3D 뷰와 수치가 항상 일치합니다.
    """
    # TODO: 파이프라인 연결
    #   1. avatar_id 로 body_bucket 조회
    #   2. {garment}_{size}__{bucket}.glb / .json 파일 존재 확인
    #      없으면 FITTING_NOT_AVAILABLE (배치 실패율 약 28%)
    #   3. 여유량 JSON 을 읽어 판정 결과 생성
    stub = FittingResponse(
        glb_url="https://cdn.closer.xxx/draped/shirt_M__H1B2.glb",
        fit_report=[
            {"part": "shoulder_width", "label": "어깨", "ease": -2.1,
             "verdict": "tight", "color": "#C0392B"},
            {"part": "chest_circ", "label": "가슴", "ease": 1.8,
             "verdict": "snug", "color": "#D68910"},
            {"part": "waist_circ", "label": "허리", "ease": 7.4,
             "verdict": "good", "color": "#27AE60"},
        ],
        overall="unfit",
        recommended_size="L",
        message="어깨가 넓고 허리가 얇은 체형입니다. M은 어깨가 2.1cm 부족합니다.",
    )
    return ApiResponse.ok(stub)
