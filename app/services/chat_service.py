"""챗봇 스트리밍.

OpenAI 응답을 SSE 이벤트로 바꿔 흘려보냅니다.

    data: {"delta": "어깨가 "}
    data: {"done": true}

키가 없으면 고정 문구를 같은 형식으로 흘립니다. 통합을 먼저 확인하기 위한
개발용 경로입니다 — 프론트는 실제 응답과 구분할 수 없으므로, 키가 들어오기
전에도 delta 누적 · 문장 단위 TTS · 에러 분기를 다 시험할 수 있습니다.
"""

import json
import logging
from typing import AsyncIterator

from app.core.config import settings
from app.models.chat import ChatRequest

logger = logging.getLogger(__name__)

# 폭주 방지 상한입니다. 길이 제어 수단이 아닙니다 — 길이는 프롬프트로 잡습니다.
MAX_TOKENS = 300

# 프롬프트 3부 조립은 별건입니다. 여기는 자리만 잡아둡니다.
# 의류 지식과 발화 규칙, fit_context · profile 풀어쓰기가 들어올 곳입니다.
_SYSTEM_PROMPT = (
    "당신은 의류 사이즈를 상담하는 한국어 도우미입니다.\n"
    "2~3문장, 120자 이내로 답하세요. 음성으로 읽히므로 길면 듣기 어렵습니다.\n"
    "받은 수치만 인용하고 없는 수치는 만들지 마세요."
)

# 키가 없을 때 흘리는 고정 응답.
# 문장 부호로 끊어 두어 프론트의 문장 단위 TTS 큐를 시험할 수 있게 합니다.
_FALLBACK_CHUNKS = (
    "지금은 챗봇 키가 설정되지 않아 ",
    "미리 준비된 문장을 보내고 있습니다. ",
    "연결은 정상입니다.",
)


def sse(payload: dict) -> str:
    """SSE 한 줄로 감쌉니다.

    ensure_ascii=False 로 한글을 그대로 보냅니다. 이스케이프하면 바이트가
    약 3배로 늘어 첫 글자가 그만큼 늦게 도착합니다.

    본문에 실제 개행이 들어가면 SSE 프레임이 그 자리에서 끊깁니다.
    json.dumps 가 개행을 \\n 으로 escape 하므로 그런 일은 생기지 않습니다.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_messages(request: ChatRequest) -> list[dict]:
    """OpenAI 에 보낼 메시지 목록.

    프롬프트 3부 조립에서 fit_context · profile 을 시스템 프롬프트로 풀어
    넣습니다. 지금은 대화만 넘깁니다.
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages += [{"role": turn.role, "content": turn.content} for turn in request.history]
    messages.append({"role": "user", "content": request.message})
    return messages


async def _fallback_events() -> AsyncIterator[str]:
    for chunk in _FALLBACK_CHUNKS:
        yield sse({"delta": chunk})
    yield sse({"done": True})


async def _openai_events(request: ChatRequest) -> AsyncIterator[str]:
    """OpenAI 스트리밍을 SSE 로 중계합니다.

    연결과 첫 토큰까지는 이 함수 밖에서 기다립니다(open_stream 참고).
    그래서 인증 실패나 네트워크 오류는 헤더가 나가기 전에 드러납니다.

    한 글자라도 보낸 뒤 끊기면 상태 코드를 바꿀 수 없으므로 스트림 안에
    error 이벤트를 넣고 끝냅니다. 프론트는 그때까지 받은 문장을 유지합니다.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_sec,
    )

    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=build_messages(request),
        stream=True,
        max_tokens=MAX_TOKENS,
    )

    sent = 0
    try:
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            sent += 1
            yield sse({"delta": delta})
    except Exception:
        logger.exception("OpenAI 스트리밍이 중단되었습니다 (보낸 조각 %d개)", sent)
        yield sse({
            "error": {
                "code": "CHAT_UPSTREAM_ERROR",
                "message": "챗봇 서버 응답에 실패했습니다",
            }
        })
        return

    yield sse({"done": True})


async def open_stream(request: ChatRequest) -> tuple[str, AsyncIterator[str]]:
    """첫 이벤트를 미리 받아 두고 나머지를 돌려줍니다.

    첫 이벤트를 여기서 기다리는 이유는 응답 헤더 때문입니다.
    ``StreamingResponse`` 를 반환한 뒤에는 이미 ``200 OK`` 와
    ``text/event-stream`` 이 나가 있어, 그 뒤에 예외를 던져도 상태 코드를
    바꿀 수 없습니다. 그러면 인증 실패 같은 서버 문제가 SSE 안으로만
    내려가고 프론트는 분기를 두 번 만들어야 합니다.

    여기서 실패하면 예외가 그대로 올라가 라우터가 일반 HTTP 로 답합니다.
    명세의 "검증은 스트림을 열기 전에 끝냅니다" 가 이 지점입니다.
    """
    events = _fallback_events() if not settings.openai_enabled else _openai_events(request)

    if not settings.openai_enabled:
        logger.warning("OPENAI_API_KEY 가 없어 고정 응답을 보냅니다")

    first = await events.__anext__()
    return first, events
