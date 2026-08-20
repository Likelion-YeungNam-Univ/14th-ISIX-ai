"""피팅 응답 스키마."""

from typing import Literal

from pydantic import BaseModel, Field

# 판정 3단계. 백엔드 FitVerdict(TIGHT / GOOD / LOOSE)와 같은 집합입니다.
#
# 예전에는 snug(0~+2cm, 주황)이 있어 4단계였습니다. 백엔드는 처음부터 3단계였고,
# 4단계로 두면 챗봇이 받지 못하는 값을 기대하고 히트맵도 색이 하나 남습니다.
# 어느 쪽으로든 통일이 필요해 백엔드 기준으로 맞췄습니다.
Verdict = Literal["loose", "good", "tight"]
Overall = Literal["fit", "unfit"]

# 히트맵 색상 — 프론트·백엔드와 같은 값입니다.
# 백엔드 FitVerdict 는 색을 blue / green / red 로 표기하고 hex 는 여기서 정합니다.
#
# 경계값은 여기 두지 않습니다. 적정 범위가 핏마다 달라
# garment/config/garment_tolerance.csv 가 단일 출처입니다.
#   슬림    둘레 -2~+3   어깨 -2~+2
#   레귤러  둘레 -4~+6   어깨 -2~+3
#   오버핏  둘레 -6~+12  어깨 -2~+5
# 판정은 절대 여유량이 아니라 편차(actual_ease − ref_ease)로 합니다.
VERDICT_COLORS: dict[str, str] = {
    "loose": "#2E86C1",  # 편차가 밴드 상한 초과 · 헐렁 (blue)
    "good": "#27AE60",   # 편차가 밴드 안 · 적정 (green)
    "tight": "#C0392B",  # 편차가 밴드 하한 미만 · 착용 어려움 (red)
}


class FitItem(BaseModel):
    part: str = Field(..., description="부위 키. measurements 와 동일")
    label: str = Field(..., description="한글 라벨")
    ease: float = Field(..., description="여유량 (cm). 음수면 낌")
    verdict: Verdict
    color: str = Field(..., description="히트맵 색상")


class FittingResponse(BaseModel):
    glb_url: str = Field(..., description="착용 메시 URL")
    fit_report: list[FitItem]
    overall: Overall
    recommended_size: str = Field(..., description="추천 사이즈")
    message: str = Field(..., description="추천 근거 문장")
