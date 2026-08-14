"""챗봇 요청 스키마.

응답은 SSE 라 스키마가 없습니다. 이벤트 형식은 명세 7.1 을 따릅니다.

    data: {"delta": "어깨가 "}
    data: {"done": true, "summary": { ... }}
    data: {"error": {"code": "...", "message": "..."}}

conversationId 는 담지 않습니다. 대화 식별과 저장은 백엔드 소관입니다.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

ChatMode = Literal["onboarding", "fitting"]
Role = Literal["user", "assistant"]

# 발화 한 번 분량입니다. 음성 입력이라 이보다 길면 STT 가 문장을 잘라 보냅니다.
MAX_MESSAGE_LEN = 500

# 요약의 "피하는것" 길이 상한. 문장이 길어지면 요약이 자유 서술로 변합니다.
MAX_PROFILE_AVOID_LEN = 20

# 판정 대상 부위. profile 의 신경쓰는부위도 같은 키를 씁니다.
# 한글 라벨("어깨")로 두면 프롬프트가 fit_report 의 shoulder_width 와
# 연결하지 못해 "어깨 여유가 2cm" 같은 문장을 만들 수 없습니다.
Part = Literal["shoulder_width", "chest_circ", "waist_circ", "hip_circ"]


class Turn(BaseModel):
    role: Role
    content: str


class FitPart(BaseModel):
    """부위 하나의 판정. 백엔드가 계산해서 보냅니다.

    AI 는 계산하지 않고 인용만 합니다. 두 곳에서 계산하면 화면 리포트와
    챗봇 답변이 다른 값을 말할 수 있습니다.
    """

    part: str
    actual_ease: float = Field(..., description="실제 여유 = 의류 치수 - 아바타 치수")
    ref_ease: float = Field(..., description="이 옷이 의도한 목표 여유")
    deviation: float = Field(..., description="편차 = 실제 여유 - 목표 여유. 판정 기준")
    verdict: Literal["loose", "good", "tight"]


class Profile(BaseModel):
    """이전 대화에서 뽑은 요약.

    항목을 고정합니다. 자유 서술을 허용하면 매번 다른 값을 뽑아 와
    비용과 품질이 같이 무너집니다.
    """

    용도: Optional[Literal["출근", "데이트", "운동", "일상"]] = None
    신경쓰는부위: list[Part] = Field(default_factory=list)
    선호핏: Optional[Literal["슬림", "레귤러", "오버핏"]] = None
    피하는것: Optional[str] = Field(None, max_length=MAX_PROFILE_AVOID_LEN)


class PastFitting(BaseModel):
    """지난 피팅 한 건. 비교 발화의 근거입니다."""

    garment_id: str
    size: str
    wearable: bool
    tight_parts: list[str] = Field(default_factory=list)
    recommended_size: Optional[str] = None


class FitContext(BaseModel):
    """판정 근거 뭉치. 백엔드가 DB 에서 조립해 보냅니다.

    body_type 은 받지 않습니다. 격자 12구간 전부가 ±4cm 계측 오차에 30% 이상
    타입이 바뀌어, 챗봇이 말로 단정하면 화면 문구보다 강하게 남습니다.
    """

    measurements: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    garment_id: Optional[str] = None
    size: Optional[str] = None
    unavailable_reason: Optional[str] = None
    recommended_size: Optional[str] = None
    fit: Optional[str] = None
    fit_report: list[FitPart] = Field(default_factory=list)

    # 개인화. 첫 대화면 profile 이 없고 past_fittings 는 빈 배열입니다.
    profile: Optional[Profile] = None
    past_fittings: list[PastFitting] = Field(default_factory=list)


class ChatRequest(BaseModel):
    mode: ChatMode
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LEN)

    # 백엔드가 잘라서 보냅니다. AI 는 받은 만큼만 씁니다.
    history: list[Turn] = Field(default_factory=list)

    # mode=onboarding 이면 보내지 않습니다. 서버가 아는 치수가 없는 상태입니다.
    fit_context: Optional[FitContext] = None
