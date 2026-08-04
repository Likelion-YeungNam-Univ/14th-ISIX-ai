"""피팅 응답 스키마."""

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["loose", "good", "snug", "tight"]
Overall = Literal["fit", "snug", "unfit"]

# 히트맵 색상 — 프론트와 동일한 값을 사용합니다
VERDICT_COLORS: dict[str, str] = {
    "loose": "#2E86C1",  # +8cm 이상 · 헐렁
    "good": "#27AE60",   # +2 ~ +8cm · 적정
    "snug": "#D68910",   # 0 ~ +2cm · 타이트
    "tight": "#C0392B",  # 0cm 미만 · 착용 불가
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
