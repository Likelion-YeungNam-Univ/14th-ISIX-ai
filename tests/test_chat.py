"""챗봇 엔드포인트 테스트.

앱 전체(app.main)를 띄우지 않고 챗 라우터만 얹어 확인합니다. main 을 import
하면 SMPL-X(약 830MB)와 mediapipe 를 로드해 CI 에서 수 분이 걸립니다.
챗 경로는 그 모델을 쓰지 않습니다.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
import app.core.config as config_module
from app.core.exceptions import ClosrException, closr_exception_handler
import app.services.chat_service as chat_service
from app.models.chat import ChatRequest, Turn
from app.routers import chat


def build_client(**overrides) -> TestClient:
    """설정을 갈아끼운 클라이언트.

    Settings 가 lru_cache 로 감싸여 있어 모듈 전역을 직접 교체합니다.
    """
    settings = Settings(_env_file=None, **overrides)
    config_module.settings = settings
    chat_service.settings = settings

    app = FastAPI()
    app.add_exception_handler(ClosrException, closr_exception_handler)
    app.include_router(chat.router)
    return TestClient(app)


def events(response) -> list[dict]:
    """SSE 응답을 이벤트 목록으로 바꿉니다."""
    out = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


FIT_CONTEXT = {
    "measurements": {"shoulder_width": 45.6, "chest_circ": 85.3},
    "garment_id": "shirt_slim",
    "size": "s",
    "fit": "슬림",
    "fit_report": [
        {"part": "shoulder_width", "actual_ease": -10.5, "ref_ease": -7.9,
         "deviation": -2.6, "verdict": "tight"},
    ],
}


class TestFallback:
    """키가 없을 때. 통합을 먼저 확인하기 위한 개발용 경로입니다."""

    def test_streams_fixed_text_as_sse(self):
        client = build_client(openai_api_key="")

        response = client.post("/api/chat", json={"mode": "onboarding", "message": "안녕하세요"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        parsed = events(response)
        assert all("delta" in event for event in parsed[:-1])
        assert parsed[-1] == {"done": True}

    def test_sends_no_buffering_header(self):
        # 프록시가 응답을 모아 두면 delta 가 완료 시점에 한꺼번에 도착합니다.
        # 스트리밍이 통째로 죽는데 로컬에서는 드러나지 않습니다.
        client = build_client(openai_api_key="")

        response = client.post("/api/chat", json={"mode": "onboarding", "message": "안녕"})

        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers["cache-control"] == "no-cache"

    def test_accepts_fit_context_with_personalization(self):
        client = build_client(openai_api_key="")
        body = {
            "mode": "fitting",
            "message": "이 셔츠 S 맞을까요?",
            "history": [{"role": "user", "content": "안녕"}],
            "fit_context": {
                **FIT_CONTEXT,
                "profile": {"용도": "출근", "신경쓰는부위": ["shoulder_width"]},
                "past_fittings": [{"garment_id": "shirt_slim", "size": "m",
                                   "wearable": False, "tight_parts": ["shoulder_width"]}],
            },
        }

        response = client.post("/api/chat", json=body)

        assert response.status_code == 200
        assert events(response)[-1] == {"done": True}


class TestValidationBeforeStream:
    """스트림을 열기 전에 끝내야 하는 검사.

    헤더가 나간 뒤에는 상태 코드를 바꿀 수 없어, 늦게 검사하면 프론트가
    같은 실패를 HTTP 와 SSE 두 곳에서 처리해야 합니다.
    """

    def test_fitting_without_context_is_rejected_as_http(self):
        client = build_client(openai_api_key="")

        response = client.post("/api/chat", json={"mode": "fitting", "message": "맞나요?"})

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["code"] == "CHAT_CONTEXT_REQUIRED"

    def test_onboarding_without_context_is_allowed(self):
        # 온보딩은 서버가 아는 치수가 없는 화면입니다.
        client = build_client(openai_api_key="")

        response = client.post("/api/chat", json={"mode": "onboarding", "message": "어떻게 써요?"})

        assert response.status_code == 200

    @pytest.mark.parametrize("message", ["", "가" * 501])
    def test_message_length_is_bounded(self, message):
        client = build_client(openai_api_key="")

        response = client.post("/api/chat", json={"mode": "onboarding", "message": message})

        assert response.status_code == 422

    def test_upstream_failure_before_first_token_is_http_not_sse(self):
        # 인증 실패는 첫 토큰 전에 드러납니다. 이때 SSE 로 내리면
        # 프론트가 200 을 받고 나서야 실패를 알게 됩니다.
        client = build_client(openai_api_key="sk-invalid", openai_timeout_sec=5)

        response = client.post("/api/chat", json={"mode": "onboarding", "message": "안녕"})

        assert response.status_code == 502
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["code"] == "CHAT_UPSTREAM_ERROR"


class TestSseFraming:
    def test_korean_is_not_escaped(self):
        # 이스케이프하면 바이트가 약 3배로 늘어 첫 글자가 그만큼 늦게 도착합니다.
        line = chat_service.sse({"delta": "어깨가 "})

        assert '"어깨가 "' in line
        assert "\\u" not in line

    def test_frame_ends_with_blank_line(self):
        line = chat_service.sse({"done": True})

        assert line.startswith("data: ")
        assert line.endswith("\n\n")

    def test_newline_in_content_does_not_break_frame(self):
        # 본문에 실제 개행이 들어가면 SSE 프레임이 그 자리에서 끊깁니다.
        line = chat_service.sse({"delta": "첫 줄\n둘째 줄"})

        assert line.count("\n") == 2
        assert "\\n" in line


class TestMessageAssembly:
    def test_order_is_system_history_then_user(self):
        request = ChatRequest(
            mode="onboarding",
            message="지금 질문",
            history=[Turn(role="user", content="이전 질문"),
                     Turn(role="assistant", content="이전 답변")],
        )

        messages = chat_service.build_messages(request)

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[-1]["content"] == "지금 질문"

    def test_system_prompt_bounds_length(self):
        # 한국어 1024토큰은 TTS 로 2~4분입니다. 길이는 프롬프트로 제어합니다.
        request = ChatRequest(mode="onboarding", message="안녕")

        system = chat_service.build_messages(request)[0]["content"]

        assert "120자" in system
