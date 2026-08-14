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

_INSTRUCTION = f"""대화에서 아래 네 항목만 뽑아 JSON 으로 답하세요.

- 용도: 출근 | 데이트 | 운동 | 일상 | null
- 신경쓰는부위: shoulder_width | chest_circ | waist_circ | hip_circ 의 배열. 없으면 []
- 선호핏: 슬림 | 레귤러 | 오버핏 | null
- 피하는것: {MAX_PROFILE_AVOID_LEN}자 이내 문자열 | null

규칙
- 사용자가 직접 말한 것만 담으세요. 추측하지 마세요.
- 근거가 없으면 null 이나 [] 로 두세요. 비워 두는 것이 틀린 값보다 낫습니다.
- 목록에 없는 값을 만들지 마세요.
- JSON 만 출력하세요. 설명을 붙이지 마세요."""


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
        return Profile(**parsed).model_dump()
    except Exception:
        logger.warning("요약 항목이 규격에 맞지 않습니다: %s", text[:200])
        return None


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
