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
    garment_id: str = Query(..., description="의류 ID (예: tshirt_basic)"),
    size: str = Query("m", pattern="^[sml]$", description="사이즈 (소문자)"),
) -> ApiResponse[FittingResponse]:
    """아바타에 의류를 착용한 결과를 조회합니다.

    파일명 규칙 — 백엔드가 이 문자열로 파일을 조회합니다.
        착용 메시   {garment}_{size}__{bucket}.glb   tshirt_basic_m__H1B2.glb
        여유량      {garment}_{size}__{bucket}.json  tshirt_basic_m__H1B2.json

    사이즈는 소문자입니다. 의류 배치가 이 규칙으로 213개를 이미 만들어 두었고,
    대문자로 바꾸려면 재생성에 두 시간이 걸려 소문자로 통일했습니다.

    여유량은 GLB 와 같은 배치에서 생성되므로 3D 뷰와 수치가 항상 일치합니다.
    """
    # TODO: 파이프라인 연결
    #   1. avatar_id 로 body_bucket 조회
    #   2. {garment}_{size}__{bucket}.glb / .json 파일 존재 확인
    #      없으면 FITTING_NOT_AVAILABLE (배치 실패 7/216, 약 3%)
    #      실패 7건 중 3건은 물리적으로 착용이 불가능한 조합이라
    #      미리보기 없이 판정 결과만 내려준다.
    #   3. 여유량 JSON 을 읽어 판정 결과 생성
    stub = FittingResponse(
        glb_url="https://cdn.closr.xxx/draped/tshirt_basic_m__H1B2.glb",
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
