"""대화 요약 추출 테스트.

여기서 지키려는 것이 둘입니다. **잘못된 요약을 저장하지 않는지**와
**실패해도 대화가 정상인지**입니다. 틀린 요약을 넣으면 다음 대화에서 챗봇이
사실이 아닌 것을 인용합니다.
"""

import json

import pytest

from app.services import chat_summary
from app.services.chat_summary import _parse, _transcript


class TestParse:
    def test_accepts_valid_summary(self):
        raw = json.dumps({"용도": "출근", "신경쓰는부위": ["shoulder_width"],
                          "선호핏": None, "피하는것": None}, ensure_ascii=False)

        assert _parse(raw) == {"용도": "출근", "신경쓰는부위": ["shoulder_width"],
                               "선호핏": None, "피하는것": None}

    def test_strips_code_fence(self):
        # 모델이 코드 블록으로 감싸 보내는 경우가 있습니다.
        raw = '```json\n{"용도": "데이트", "신경쓰는부위": []}\n```'

        assert _parse(raw)["용도"] == "데이트"

    def test_rejects_non_json(self):
        assert _parse("요약: 출근용 셔츠를 찾고 있습니다") is None

    def test_rejects_value_outside_the_list(self):
        # 목록에 없는 값을 저장하면 다음 대화에서 없는 취향을 인용합니다.
        raw = json.dumps({"용도": "결혼식", "신경쓰는부위": []}, ensure_ascii=False)

        assert _parse(raw) is None

    def test_rejects_korean_part_label(self):
        # 한글 라벨을 저장하면 프롬프트가 fit_report 와 연결하지 못합니다.
        raw = json.dumps({"신경쓰는부위": ["어깨"]}, ensure_ascii=False)

        assert _parse(raw) is None

    def test_rejects_long_avoid_text(self):
        # 문장이 길어지면 요약이 자유 서술로 변합니다.
        raw = json.dumps({"피하는것": "가" * 21}, ensure_ascii=False)

        assert _parse(raw) is None

    def test_rejects_array_at_root(self):
        assert _parse('[{"용도": "출근"}]') is None


class TestTranscript:
    def test_includes_last_turn(self):
        # 마지막 발화와 답변이 빠지면 방금 말한 취향을 놓칩니다.
        text = _transcript(
            [{"role": "user", "content": "안녕"}], "출근용 셔츠 찾아요", "네, 도와드릴게요")

        assert "user: 출근용 셔츠 찾아요" in text
        assert "assistant: 네, 도와드릴게요" in text

    def test_survives_empty_answer(self):
        # 스트림이 중간에 끊겨 답변이 비었을 수 있습니다.
        text = _transcript([], "질문만 있음", "")

        assert text == "user: 질문만 있음"


class TestExtract:
    @pytest.mark.asyncio
    async def test_returns_none_without_key(self, monkeypatch):
        # 키가 없으면 호출하지 않습니다. 고정 응답 단계에서도 대화는 정상입니다.
        monkeypatch.setattr(chat_summary.settings, "openai_api_key", "", raising=False)

        assert await chat_summary.extract([], "질문", "답변") is None
