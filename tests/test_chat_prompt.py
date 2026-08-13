"""시스템 프롬프트 조립 테스트.

여기서 지키려는 것이 셋입니다. **개인화가 비어도 동작하는지**,
**금지 규칙이 빠지지 않는지**, **부위·사이즈 표기가 규칙대로인지**입니다.
"""

from app.models.chat import ChatRequest
from app.services.chat_prompt import build_system_prompt

FIT_REPORT = [
    {"part": "shoulder_width", "actual_ease": -11.0, "ref_ease": -7.9,
     "deviation": -3.1, "verdict": "tight"},
    {"part": "chest_circ", "actual_ease": 6.0, "ref_ease": 8.0,
     "deviation": -2.0, "verdict": "good"},
]


def fitting(**context) -> ChatRequest:
    base = {"measurements": {}, "warnings": [], "garment_id": "shirt_slim",
            "size": "s", "fit": "슬림", "recommended_size": "m",
            "fit_report": FIT_REPORT}
    base.update(context)
    return ChatRequest(mode="fitting", message="이건 어때요?", fit_context=base)


class TestAlwaysPresent:
    def test_length_limit_is_stated(self):
        # 한국어 1024토큰은 TTS 로 2~4분입니다. 길이는 프롬프트로 제어합니다.
        assert "120자" in build_system_prompt(fitting())

    def test_forbids_color_talk(self):
        # 잘 맞는 옷도 빨강이 29% 나옵니다. 색으로는 사이즈를 가릴 수 없습니다.
        prompt = build_system_prompt(fitting())
        assert "색으로 사이즈를 말하지" in prompt

    def test_forbids_body_type_talk(self):
        # 격자 12구간 전부가 ±4cm 오차에 30% 이상 타입이 바뀝니다.
        assert "체형 유형" in build_system_prompt(fitting())

    def test_forbids_softening_tight(self):
        # 화면은 "착용이 어렵습니다" 로 표시합니다. 챗봇이 "다소" 라고 하면 어긋납니다.
        assert "다소 낍니다" in build_system_prompt(fitting())

    def test_states_comparison_threshold(self):
        # 0.3cm 차이를 "더 여유" 라고 하면 계측 오차보다 작은 값을 근거로 삼습니다.
        assert "1cm 미만" in build_system_prompt(fitting())

    def test_states_priority_order(self):
        assert "뒤에서부터" in build_system_prompt(fitting())


class TestOnboarding:
    def test_has_no_measurements_block(self):
        prompt = build_system_prompt(ChatRequest(mode="onboarding", message="어떻게 써요?"))

        assert "아바타를 먼저 만들어주세요" in prompt
        assert "[지금 보고 있는 옷]" not in prompt

    def test_fitting_without_context_falls_back_to_onboarding(self):
        # 라우터가 막지만, 프롬프트도 치수를 지어내지 않아야 합니다.
        prompt = build_system_prompt(ChatRequest(mode="fitting", message="맞나요?"))

        assert "치수를 추측하지 마세요" in prompt


class TestPersonalization:
    def test_profile_block_is_omitted_when_absent(self):
        # 첫 대화에는 요약이 없습니다. 이 덩어리가 빠져도 답변은 나가야 합니다.
        prompt = build_system_prompt(fitting(profile=None))

        assert "[지난 대화에서 알게 된 것]" not in prompt
        assert "[지금 보고 있는 옷]" in prompt

    def test_profile_uses_korean_part_labels(self):
        # 프롬프트에는 사람이 읽는 말로, 판정 근거는 영문 키로 받습니다.
        prompt = build_system_prompt(
            fitting(profile={"용도": "출근", "신경쓰는부위": ["shoulder_width"]}))

        assert "신경 쓰는 부위: 어깨" in prompt
        assert "shoulder_width" not in prompt.split("[지금 보고 있는 옷]")[0]

    def test_profile_tells_to_quote_once(self):
        prompt = build_system_prompt(
            fitting(profile={"용도": "출근", "선호핏": "오버핏"}))

        assert "하나만 골라 한 번 인용" in prompt

    def test_empty_profile_object_is_omitted(self):
        # 요약이 만들어졌지만 모든 항목이 null 인 경우입니다.
        prompt = build_system_prompt(fitting(profile={}))

        assert "[지난 대화에서 알게 된 것]" not in prompt

    def test_past_fittings_block_is_omitted_when_empty(self):
        assert "[지난번에 본 옷]" not in build_system_prompt(fitting(past_fittings=[]))

    def test_past_fittings_shows_tight_parts_in_korean(self):
        prompt = build_system_prompt(fitting(past_fittings=[
            {"garment_id": "shirt_slim", "size": "m", "wearable": False,
             "tight_parts": ["shoulder_width"]}]))

        assert "shirt_slim M: 어깨 꽉 낌" in prompt


class TestFitContext:
    def test_sizes_are_uppercase(self):
        # 사이즈는 대문자로 말합니다. R2 파일명만 소문자입니다.
        prompt = build_system_prompt(fitting(size="s", recommended_size="m"))

        assert "shirt_slim S" in prompt
        assert "추천 사이즈: M" in prompt

    def test_report_uses_deviation_and_korean_parts(self):
        prompt = build_system_prompt(fitting())

        assert "어깨: 편차 -3.1cm (꽉 낌)" in prompt
        assert "가슴: 편차 -2.0cm (적정)" in prompt

    def test_unavailable_reason_is_framed_as_preview_only(self):
        # 미리보기가 없다는 것과 못 입는다는 것은 다릅니다.
        prompt = build_system_prompt(fitting(unavailable_reason="TOO_SMALL"))

        assert "3D 미리보기가 없습니다" in prompt
        assert "판정은 위 수치대로 있으니" in prompt

    def test_warnings_ask_for_a_caveat(self):
        prompt = build_system_prompt(
            fitting(warnings=["팔이 몸통에 붙어 있습니다"]))

        assert "어깨 수치만으로 단정하지 말고" in prompt

    def test_measurements_use_korean_labels(self):
        prompt = build_system_prompt(
            fitting(measurements={"shoulder_width": 45.6, "chest_circ": 85.3}))

        assert "어깨 45.6cm" in prompt
        assert "가슴 85.3cm" in prompt
