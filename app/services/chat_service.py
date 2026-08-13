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
from app.services import chat_prompt, chat_summary

logger = logging.getLogger(__name__)

# 폭주 방지 상한입니다. 길이 제어 수단이 아닙니다 — 길이는 프롬프트로 잡습니다.
MAX_TOKENS = 300

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

    시스템 프롬프트는 고정 지시 + 개인화 + 근거 세 덩어리로 조립합니다
    (:mod:`app.services.chat_prompt`).
    """
    messages = [{"role": "system", "content": chat_prompt.build_system_prompt(request)}]
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
    answer: list[str] = []
    try:
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            sent += 1
            answer.append(delta)
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

    yield sse(await _done_payload(request, "".join(answer)))


async def _done_payload(request: ChatRequest, answer: str) -> dict:
    """마지막 이벤트. 대화 요약을 함께 실어 보냅니다.

    요약을 여기서 만드는 이유는 지연이 보이지 않기 때문입니다. 답변은 이미
    화면에 다 떴고 마지막 문장 TTS 가 재생 중이라, 1초쯤 걸려도 사용자는
    모릅니다. 백그라운드로 돌리면 실패해도 아무에게도 안 보입니다.

    onboarding 은 요약하지 않습니다. 치수도 취향도 아직 나오지 않은 단계라
    뽑을 것이 없고, 호출만 낭비됩니다.
    """
    payload: dict = {"done": True}

    if request.mode == "onboarding":
        return payload

    history = [{"role": turn.role, "content": turn.content} for turn in request.history]
    summary = await chat_summary.extract(history, request.message, answer)
    if summary is not None:
        # 백엔드만 소비합니다. 프론트로 넘기지 않습니다.
        payload["summary"] = summary

    return payload


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
