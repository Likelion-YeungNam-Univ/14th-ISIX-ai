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
    # 판정 대상 부위는 의류마다 다르다. garment_tolerance.csv 에 정의된 것만 넣는다.
    # 티셔츠·셔츠는 어깨·가슴뿐이고 허리는 대상이 아니다. 대상이 아닌 부위를
    # 넣으면 기준 여유가 없어 편차를 계산할 수 없고, 챗봇이 근거 없는 수치를
    # 말하게 된다.
    #
    # 아래 수치는 tshirt_basic M 의 실제 기준 여유를 쓴 것이다.
    #   어깨  ref -9.0,  레귤러 밴드 -2~+3  ->  편차 -2.6 이면 tight
    #   가슴  ref 14.0,  레귤러 밴드 -4~+6  ->  편차 -1.5 이면 good
    stub = FittingResponse(
        glb_url="https://cdn.closr.xxx/draped/tshirt_basic_m__H1B2.glb",
        fit_report=[
            {"part": "shoulder_width", "label": "어깨", "ease": -11.6,
             "verdict": "tight", "color": "#C0392B"},
            {"part": "chest_circ", "label": "가슴", "ease": 12.5,
             "verdict": "good", "color": "#27AE60"},
        ],
        overall="unfit",
        recommended_size="l",
        message="M 은 어깨가 기준보다 2.6cm 부족합니다. L 을 권합니다.",
    )
    return ApiResponse.ok(stub)
