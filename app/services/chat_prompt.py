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

from pathlib import Path
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
# 의류 파트가 작성한 상담 규칙입니다. 코드에 문장을 박아 넣지 않고 파일로 둡니다.
# 문구를 다듬는 사람과 코드를 고치는 사람이 다르고, 문장이 바뀔 때마다 파이썬
# 파일을 건드리면 리뷰에서 규칙 변경과 로직 변경이 섞입니다.
#
# 파일이 없으면 기동에 실패시킵니다. 규칙 없이 뜨면 길이 제한도 금지 사항도
# 없는 챗봇이 되는데, 응답은 정상으로 보여서 발견이 늦습니다.
_PROMPT_PATH = Path(__file__).with_name("garment_prompt.txt")

try:
    GARMENT_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()
except OSError as exc:  # pragma: no cover - 배포 누락
    raise RuntimeError(f"{_PROMPT_PATH.name} 을 읽을 수 없습니다") from exc

if not GARMENT_PROMPT:
    raise RuntimeError(f"{_PROMPT_PATH.name} 이 비어 있습니다")


def _topic(noun: str) -> str:
    """주제 조사. 받침이 있으면 "은", 없으면 "는" 입니다.

    한쪽으로 고정하면 부위 넷 중 하나는 반드시 틀립니다 — 가슴은 · 어깨는.
    """
    if not noun:
        return "는"
    last = noun[-1]
    if not ("가" <= last <= "힣"):
        return "는"
    return "은" if (ord(last) - 0xAC00) % 28 else "는"


def _profile_block(profile: Optional[Profile], judged: set[str]) -> str:
    """이전 대화 요약. 첫 대화면 빈 문자열입니다.

    ``judged`` 는 이번 옷에서 실제로 판정한 부위입니다. **거기에 없는 부위를
    사용자가 신경 쓴다고 적어 두면 모델이 그 부위의 판정을 지어냅니다.**
    어깨가 신경 쓰인다고 말한 사용자가 슬랙스를 고르면 판정에는 허리만 있는데
    "어깨는 잘 맞습니다" 를 덧붙이는 식입니다.

    그래서 판정에 없는 부위는 말하지 말라고 못박습니다. 항목 자체를 지우지는
    않습니다 — 용도나 선호 핏은 그대로 쓸 수 있고, 부위도 사용자가 직접 물으면
    답해야 합니다.
    """
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

    unjudged = [PART_LABELS.get(p, p) for p in profile.신경쓰는부위 if p not in judged]
    if unjudged:
        listed = " · ".join(unjudged)
        warning = (
            f"\n{listed}{_topic(listed)} 이번 옷의 판정에 없습니다. "
            "그 부위가 맞는지 여부를 말하지 마십시오."
        )
    else:
        warning = ""

    return (
        "\n[지난 대화에서 알게 된 것]\n"
        + "\n".join(lines)
        + warning
        + "\n이 중 하나만 골라 한 번 인용하세요. 두 개를 다 인용하면 말투가 부자연스러워집니다."
    )


def _past_fittings_block(past: list[PastFitting], has_garment: bool) -> str:
    """지난 피팅. 비교 발화의 근거입니다.

    **비교할 대상이 있을 때만 비교를 지시합니다.** 옷을 고르지 않은 상태에서
    "지금 옷과 비교하세요" 를 남겨 두면 모델이 지난 옷을 지금 옷처럼 말합니다.
    지시가 이행 불가능하면 모델은 지시를 버리는 대신 전제를 만들어 냅니다.
    """
    if not past:
        return ""

    lines = []
    for fitting in past:
        tight = " · ".join(PART_LABELS.get(p, p) for p in fitting.tight_parts)
        state = f"{tight} 꽉 낌" if tight else "전 부위 적정"
        lines.append(f"- {fitting.garment_id} {fitting.size.upper()}: {state}")

    closing = (
        "\n지금 옷과 수치 차이가 1cm 이상일 때만 비교하세요."
        if has_garment
        else "\n지금 고른 옷이 없습니다. 먼저 꺼내 말하지 마시고, "
             "사용자가 그 옷을 물었을 때만 답하세요."
    )
    return "\n[지난번에 본 옷]\n" + "\n".join(lines) + closing


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
    # 옷이 없는데 "[지금 보고 있는 옷]" 을 붙이면 모델이 있다고 믿고 설명합니다.
    # 그러면 지난 피팅이나 예시의 옷을 현재 옷처럼 말하게 됩니다.
    if not context.garment_id:
        # 치수가 있는데도 "아바타를 만들어 주세요" 라고 답한 사례가 있었습니다.
        # 옷이 없는 것과 아바타가 없는 것을 섞은 것이라 그 둘을 갈라 둡니다.
        lines = ["\n[사용자 치수]",
                 "- 아직 옷을 고르지 않았습니다. 옷에 대한 판정을 말하지 마십시오.",
                 "- 치수는 아래에 이미 있습니다. 아바타를 만들라거나 측정이 필요하다고"
                 " 말하지 마십시오.",
                 # "어떤 옷인지 말씀해 주시면" 처럼 사용자에게 옷을 설명하라고
                 # 답한 사례가 있었습니다. 옷은 화면에서 고르는 것입니다.
                 "- \"화면에서 옷을 고르시면 사이즈를 봐 드리겠습니다\" 처럼"
                 " 안내하세요. 옷 이름을 말해 달라고 하지 마십시오."]
    else:
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
    blocks = [GARMENT_PROMPT]

    context = request.fit_context
    if request.mode == "onboarding" or context is None:
        # 규칙 본문에 onboarding 절이 있습니다. 여기서는 지금이 그 상태라는
        # 것만 알립니다. fit_context 를 빈 값으로 넣으면 모델이 0 을 수치로
        # 읽어 없는 치수를 말하게 됩니다.
        blocks.append("\n[지금 상태]\nmode 는 onboarding 입니다. 치수와 판정 결과가 없습니다.")
        return "\n".join(blocks)

    judged = {part.part for part in context.fit_report}
    blocks.append(_profile_block(context.profile, judged))
    blocks.append(_fit_context_block(context))
    blocks.append(_past_fittings_block(
        context.past_fittings, has_garment=bool(context.garment_id)))

    return "\n".join(block for block in blocks if block)
