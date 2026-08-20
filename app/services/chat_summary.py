"""대화 요약 추출.

스트림이 끝난 뒤 ``done`` 직전에 한 번 호출합니다. 백그라운드 작업을 두지
않습니다 — 비동기로 돌리면 실패해도 아무에게도 안 보이고, 조용히 안 되는
상태로 남습니다.

이 1초는 사용자에게 보이지 않습니다. 답변은 이미 화면에 다 떴고 마지막 문장
TTS 가 재생 중입니다.

**항목을 고정합니다.** 자유 서술을 허용하면 매번 다른 값을 뽑아 와 비용과
품질이 같이 무너집니다.
"""

import json
import logging
from typing import Optional

from app.core.config import settings
from app.models.chat import MAX_PROFILE_AVOID_LEN, Profile

logger = logging.getLogger(__name__)

# 요약은 짧습니다. 넉넉히 줘도 4항목이면 100토큰을 넘지 않습니다.
MAX_TOKENS = 200

# **근거가 없으면 키를 아예 빼라고 합니다.** null 을 넣으라고 하면 모델이 JSON
# null 대신 문자열 "null" 을 보낼 때가 있고, 그러면 항목마다 다르게, 그리고
# 둘 다 조용히 실패합니다(#23). 키 생략은 타입 구분보다 훨씬 안정적입니다.
# Profile 네 항목 모두 기본값이 있어 빠진 키는 그대로 빈 값이 됩니다.
_INSTRUCTION = f"""대화에서 아래 네 항목만 뽑아 JSON 으로 답하세요.

- 용도: 출근 | 데이트 | 운동 | 일상
- 신경쓰는부위: shoulder_width | chest_circ | waist_circ | hip_circ 의 배열
- 선호핏: 슬림 | 레귤러 | 오버핏
- 피하는것: {MAX_PROFILE_AVOID_LEN}자 이내 문자열

규칙
- 사용자가 직접 말한 것만 담으세요. 추측하지 마세요.
- **근거가 없는 항목은 키를 아예 넣지 마세요.** null 이나 "null" 을 쓰지 마세요.
  네 항목 다 근거가 없으면 {{}} 를 출력하세요.
- 목록에 없는 값을 만들지 마세요.
- JSON 만 출력하세요. 설명을 붙이지 마세요.

예) 출근용을 찾고 어깨가 신경 쓰인다고만 말한 경우
{{"용도": "출근", "신경쓰는부위": ["shoulder_width"]}}"""


def _transcript(history: list[dict], message: str, answer: str) -> str:
    """요약에 넣을 대화. 마지막 턴까지 포함합니다."""
    lines = [f"{turn['role']}: {turn['content']}" for turn in history]
    lines.append(f"user: {message}")
    if answer:
        lines.append(f"assistant: {answer}")
    return "\n".join(lines)


def _parse(raw: str) -> Optional[dict]:
    """모델 출력을 검증합니다.

    형태가 어긋나면 ``None`` 을 돌려주고 요약을 건너뜁니다. 잘못된 요약을
    저장하면 다음 대화에서 챗봇이 사실이 아닌 것을 인용합니다.
    """
    text = raw.strip()
    # 코드 블록으로 감싸 보내는 경우가 있습니다.
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("요약이 JSON 이 아닙니다: %s", raw[:200])
        return None

    if not isinstance(parsed, dict):
        return None

    try:
        # Profile 이 값 목록까지 검사합니다. 목록에 없는 값이면 여기서 걸립니다.
        return Profile(**_nulls(parsed)).model_dump()
    except Exception:
        logger.warning("요약 항목이 규격에 맞지 않습니다: %s", text[:200])
        return None


# 모델이 "비었다" 를 표현하는 방식들. 지시문에 null 이라 적어 두면 그것을
# 문자열로 그대로 적어 보내는 경우가 있습니다.
_EMPTY = {"null", "none", "nan", "n/a", "없음", "미정", "-", ""}


def _nulls(parsed: dict) -> dict:
    """비었다는 뜻의 문자열을 진짜 ``None`` 으로 바꿉니다.

    **간헐적으로 요약이 통째로 사라지던 원인입니다.** 지시문이 ``null`` 을
    쓰라고 적어 두었더니 모델이 JSON ``null`` 대신 문자열 ``"null"`` 을 보낼
    때가 있습니다. 같은 모델이 어떤 턴은 제대로, 어떤 턴은 문자열로 보냅니다.

    그대로 검증에 넘기면 두 가지로 갈립니다. 어느 쪽도 조용합니다.

        선호핏 "null"    목록에 없는 값이라 예외 → 요약 4항목 전부 폐기
        피하는것 "null"  20자 이내 문자열이라 통과 → 다음 프롬프트에
                        "피하는 것: null" 이 그대로 주입

    앞쪽은 개인화가 사라지고 뒤쪽은 없는 취향이 생깁니다. 답변 자체는 정상이라
    화면에서는 드러나지 않고, 다음 턴에 기억을 못 하는 형태로만 보입니다.

    값을 만들어 내지는 않습니다. 비었다는 표시를 비었다고 읽을 뿐입니다.
    """
    cleaned = {}
    for key, value in parsed.items():
        if isinstance(value, str) and value.strip().lower() in _EMPTY:
            cleaned[key] = None
        elif isinstance(value, list):
            # 신경쓰는부위 에 ["null"] 로 오는 경우도 같이 걸러냅니다.
            cleaned[key] = [
                item for item in value
                if not (isinstance(item, str) and item.strip().lower() in _EMPTY)
            ]
        else:
            cleaned[key] = value

    # 배열 항목에 None 이 오면 Profile 이 거부합니다. 리스트가 아예 없는 것과
    # 같게 취급합니다.
    if cleaned.get("신경쓰는부위") is None:
        cleaned["신경쓰는부위"] = []
    return cleaned


async def extract(history: list[dict], message: str, answer: str) -> Optional[dict]:
    """대화 요약을 만듭니다. 실패하면 ``None`` 입니다.

    실패해도 대화는 정상입니다. 호출부가 ``summary`` 없이 ``done`` 을 보냅니다.
    """
    if not settings.openai_enabled:
        return None

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_sec,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": _transcript(history, message, answer)},
            ],
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception:
        # 요약이 안 돼도 답변은 이미 나갔습니다. 다음 턴에 다시 시도합니다.
        logger.exception("대화 요약에 실패했습니다")
        return None

    if not response.choices:
        return None

    summary = _parse(response.choices[0].message.content or "")
    if summary is None:
        return None

    # 전부 비었으면 저장할 것이 없습니다. 빈 요약을 넣으면 다음 대화에서
    # profile 블록이 붙었다 사라져 프롬프트가 흔들립니다.
    if not any(summary.values()):
        return None

    return summary
