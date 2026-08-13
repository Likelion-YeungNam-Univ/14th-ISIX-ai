"""시스템 프롬프트 조립.

세 덩어리를 이어 붙입니다.

    ① 고정 지시   말투 · 길이 · 금지 사항. 의류 지식이 들어갈 자리
    ② 개인화      profile — 이전 대화 요약. 없으면 이 덩어리가 빠집니다
    ③ 근거        fit_context — 치수 · 판정 · 지난 피팅

**②가 비어도 ①③만으로 동작해야 합니다.** 첫 대화에는 요약이 없고,
개인화 작업이 늦어지면 ②가 계속 비어 있게 됩니다.

수치는 만들지 않고 받은 것만 인용합니다. AI 가 자체 계산하면 화면 리포트와
챗봇 답변이 다른 값을 말할 수 있어, 정확도 항목 `사이즈 추천 일관성`(MUST)에
걸립니다.
"""

from typing import Optional

from app.models.chat import ChatRequest, FitContext, PastFitting, Profile

# 부위 키를 한글로 옮깁니다. 프롬프트에는 사람이 읽는 말로 넣고, 판정 근거는
# 영문 키로 받습니다. 한글만 주면 모델이 fit_report 와 연결하지 못합니다.
PART_LABELS = {
    "shoulder_width": "어깨",
    "chest_circ": "가슴",
    "waist_circ": "허리",
    "hip_circ": "엉덩이",
}

VERDICT_LABELS = {
    "tight": "꽉 낌",
    "good": "적정",
    "loose": "여유 있음",
}

# ── ① 고정 지시 ─────────────────────────────────────────────────────────
#
# 의류 지식(핏별 특성 · 부위별 설명)은 의류 파트가 작성합니다. 받으면
# GARMENT_KNOWLEDGE 에 넣습니다. 비어 있어도 답변은 나갑니다.
GARMENT_KNOWLEDGE = ""

BASE_INSTRUCTION = """당신은 의류 사이즈를 상담하는 한국어 도우미입니다.

[길이]
- 2~3문장, 120자 이내로 답하세요. 음성으로 읽히므로 길면 듣기 어렵습니다.
- 넘칠 것 같으면 뒤에서부터 뺍니다: ①현재 판정 ②어깨 경고 ③지난 대화 인용 ④지난 피팅 비교
- ①②는 빼지 않습니다. ④를 먼저 뺍니다.
- 어깨 경고와 지난 피팅 비교를 한 답변에 같이 넣지 마세요. 둘 다 넣으면 120자를 넘습니다.

[수치]
- 받은 수치만 인용하세요. 없는 수치를 만들지 마세요.
- 판정은 편차(deviation)로 말합니다. 실제 여유(actual_ease)는 사용자가 직접 물을 때만 씁니다.
- 두 수치의 차이가 1cm 미만이면 비교하지 마세요. 계측 오차(±4cm)보다 작습니다.
- 부위는 한글로 부릅니다. 어깨 · 가슴 · 허리 · 엉덩이.
- 사이즈는 대문자로 말합니다. S · M · L.

[금지]
- 색으로 사이즈를 말하지 마세요. 3D 히트맵 색은 사이즈를 구분하지 못합니다.
  잘 맞는 옷도 붉은 부분이 30% 정도 나옵니다.
- 체형 유형(모래시계형 등)을 말하지 마세요. 계측 오차에 쉽게 뒤집힙니다.
- "꽉 낌" 을 "다소 낍니다" 처럼 완화하지 마세요. 화면은 "착용이 어렵습니다" 로
  표시하므로 같은 상태를 다르게 말하게 됩니다. 대신 추천 사이즈를 안내하세요.
- 미리보기가 없다는 것과 못 입는다는 것은 다릅니다. 미리보기가 없어도
  판정은 있습니다."""

ONBOARDING_INSTRUCTION = """
[지금 상태]
아직 아바타가 없어 사용자의 치수를 모릅니다. 치수나 사이즈를 묻는 질문에는
"사진을 올려 아바타를 먼저 만들어주세요" 로 안내하세요. 치수를 추측하지 마세요.
서비스 설명과 촬영 안내까지만 답합니다."""


def _profile_block(profile: Optional[Profile]) -> str:
    """이전 대화 요약. 첫 대화면 빈 문자열입니다."""
    if profile is None:
        return ""

    lines = []
    if profile.용도:
        lines.append(f"- 용도: {profile.용도}")
    if profile.신경쓰는부위:
        parts = " · ".join(PART_LABELS.get(p, p) for p in profile.신경쓰는부위)
        lines.append(f"- 신경 쓰는 부위: {parts}")
    if profile.선호핏:
        lines.append(f"- 선호하는 핏: {profile.선호핏}")
    if profile.피하는것:
        lines.append(f"- 피하는 것: {profile.피하는것}")

    if not lines:
        return ""

    return (
        "\n[지난 대화에서 알게 된 것]\n"
        + "\n".join(lines)
        + "\n이 중 하나만 골라 한 번 인용하세요. 두 개를 다 인용하면 말투가 부자연스러워집니다."
    )


def _past_fittings_block(past: list[PastFitting]) -> str:
    """지난 피팅. 비교 발화의 근거입니다."""
    if not past:
        return ""

    lines = []
    for fitting in past:
        tight = " · ".join(PART_LABELS.get(p, p) for p in fitting.tight_parts)
        state = f"{tight} 꽉 낌" if tight else "전 부위 적정"
        lines.append(f"- {fitting.garment_id} {fitting.size.upper()}: {state}")

    return (
        "\n[지난번에 본 옷]\n"
        + "\n".join(lines)
        + "\n지금 옷과 수치 차이가 1cm 이상일 때만 비교하세요."
    )


def _fit_report_block(context: FitContext) -> str:
    if not context.fit_report:
        return ""

    lines = []
    for part in context.fit_report:
        label = PART_LABELS.get(part.part, part.part)
        verdict = VERDICT_LABELS.get(part.verdict, part.verdict)
        lines.append(
            f"  · {label}: 편차 {part.deviation:+.1f}cm ({verdict}), "
            f"실제 여유 {part.actual_ease:+.1f}cm / 기준 {part.ref_ease:+.1f}cm"
        )
    return "\n".join(lines)


def _fit_context_block(context: FitContext) -> str:
    """치수와 판정. 답변의 본체입니다."""
    lines = ["\n[지금 보고 있는 옷]"]

    if context.garment_id:
        size = (context.size or "").upper()
        fit = f", {context.fit}" if context.fit else ""
        lines.append(f"- 의류: {context.garment_id} {size}{fit}")
    if context.recommended_size:
        lines.append(f"- 추천 사이즈: {context.recommended_size.upper()}")

    report = _fit_report_block(context)
    if report:
        lines.append("- 부위별 판정")
        lines.append(report)

    if context.unavailable_reason:
        # 미리보기 유무입니다. 착용 가능 여부와 섞어 말하면 화면과 어긋납니다.
        lines.append(
            "- 이 사이즈는 3D 미리보기가 없습니다. "
            "판정은 위 수치대로 있으니 그것으로 답하세요."
        )

    if context.measurements:
        measured = " · ".join(
            f"{PART_LABELS.get(key, key)} {value:.1f}cm"
            for key, value in context.measurements.items()
            if key in PART_LABELS
        )
        if measured:
            lines.append(f"- 사용자 실측: {measured}")

    if context.warnings:
        # 어깨가 과대 추정된 사진입니다. 수치만으로 단정하면 안 됩니다.
        lines.append(
            "- 계측 경고가 있습니다: " + " / ".join(context.warnings)
            + "\n  어깨 수치만으로 단정하지 말고, 사진 자세에 따라 오차가 있을 수 있다고"
            " 함께 안내하세요."
        )

    return "\n".join(lines)


def build_system_prompt(request: ChatRequest) -> str:
    """세 덩어리를 이어 붙입니다."""
    blocks = [BASE_INSTRUCTION]

    if GARMENT_KNOWLEDGE.strip():
        blocks.append("\n[의류 지식]\n" + GARMENT_KNOWLEDGE.strip())

    context = request.fit_context
    if request.mode == "onboarding" or context is None:
        blocks.append(ONBOARDING_INSTRUCTION)
        return "\n".join(blocks)

    blocks.append(_profile_block(context.profile))
    blocks.append(_fit_context_block(context))
    blocks.append(_past_fittings_block(context.past_fittings))

    return "\n".join(block for block in blocks if block)
