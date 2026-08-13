"""챗봇 라우터.

백엔드가 중계합니다. 프론트는 이 주소를 직접 부르지 않습니다.

    [브라우저] ──SSE── [백엔드 /api/v1/chat] ──SSE── [여기]

음성은 브라우저가 처리합니다. STT · TTS 가 Web Speech API 라 서버로
오디오가 올라오지 않고, 이 엔드포인트는 텍스트만 주고받습니다.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.exceptions import ClosrException, ErrorCode
from app.models.chat import ChatRequest
from app.services import chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# 프록시가 응답을 모아 두면 delta 가 하나씩 오지 않고 완료 시점에 한꺼번에
# 도착합니다. 스트리밍이 통째로 죽는데 로컬에서는 드러나지 않습니다.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.post("", response_class=StreamingResponse)
async def chat(request: ChatRequest) -> StreamingResponse:
    """대화 한 턴을 스트리밍으로 답합니다.

    응답은 SSE 이고 ``conversationId`` 는 담지 않습니다. 대화 식별과 저장은
    백엔드 소관입니다.

        data: {"delta": "어깨가 "}
        data: {"done": true}

    에러는 두 경로로 나갑니다. 스트림을 열기 **전** 실패는 일반 HTTP 상태로,
    연 **뒤** 실패는 스트림 안의 ``error`` 이벤트로 갑니다. 헤더가 나간 뒤에는
    상태 코드를 바꿀 수 없기 때문입니다.
    """
    if request.mode == "fitting" and request.fit_context is None:
        # 치수 없이 사이즈를 답하면 없는 수치를 지어냅니다.
        raise ClosrException(ErrorCode.CHAT_CONTEXT_REQUIRED)

    try:
        first, rest = await chat_service.open_stream(request)
    except Exception:
        # 아직 헤더가 나가지 않았으므로 상태 코드로 알릴 수 있습니다.
        logger.exception("챗봇 스트림을 열지 못했습니다")
        raise ClosrException(ErrorCode.CHAT_UPSTREAM_ERROR)

    async def body():
        yield first
        async for event in rest:
            yield event

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
