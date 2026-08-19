"""시스템 프롬프트 조립 테스트.

여기서 지키려는 것이 셋입니다. **개인화가 비어도 동작하는지**,
**금지 규칙이 빠지지 않는지**, **부위·사이즈 표기가 규칙대로인지**입니다.
"""

from app.models.chat import ChatRequest, FitContext
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
        assert "색을 근거로 사이즈를 말하지 마십시오" in prompt

    def test_forbids_body_type_talk(self):
        # 격자 12구간 전부가 ±4cm 오차에 30% 이상 타입이 바뀝니다.
        assert "체형 유형" in build_system_prompt(fitting())

    def test_forbids_softening_tight(self):
        # 화면은 "착용이 어렵습니다" 로 표시합니다. 챗봇이 "다소" 라고 하면 어긋납니다.
        assert '"다소 낍니다" 처럼 완화하지 마십시오' in build_system_prompt(fitting())

    def test_states_comparison_threshold(self):
        # 0.3cm 차이를 "더 여유" 라고 하면 계측 오차보다 작은 값을 근거로 삼습니다.
        assert "1cm 미만" in build_system_prompt(fitting())

    def test_states_priority_order(self):
        # v5 — "뒤에서 빼기" 에서 "앞에서 쌓기" 로 바뀌었습니다. 다 쓴 뒤 줄이면
        # 먼저 쓴 것이 남고 판정 수치가 밀려 나갑니다.
        prompt = build_system_prompt(fitting())

        assert "앞에서 쌓" in prompt
        assert "수치가 하나도 없는 답변은 실패" in prompt
        assert "판정 부위를 전부" in prompt


class TestOnboarding:
    def test_has_no_measurements_block(self):
        prompt = build_system_prompt(ChatRequest(mode="onboarding", message="어떻게 써요?"))

        assert "먼저 아바타를 만들어 달라고 안내" in prompt
        assert "[지금 보고 있는 옷]" not in prompt

    def test_fitting_without_context_falls_back_to_onboarding(self):
        # 라우터가 막지만, 프롬프트도 치수를 지어내지 않아야 합니다.
        prompt = build_system_prompt(ChatRequest(mode="fitting", message="맞나요?"))

        assert "없는 수치를 지어내지 말고" in prompt


class TestPersonalization:
    def test_profile_block_is_omitted_when_absent(self):
        # 첫 대화에는 요약이 없습니다. 이 덩어리가 빠져도 답변은 나가야 합니다.
        prompt = build_system_prompt(fitting(profile=None))

        assert "[지난 대화에서 알게 된 것]" not in prompt
        assert "[지금 보고 있는 옷]" in prompt

    def test_profile_uses_korean_part_labels(self):
        # 프롬프트에는 사람이 읽는 말로 넣습니다. 규칙 본문에는 영문 키가
        # 나오지만(모델이 fit_report 와 연결해야 함), 주입되는 값은 한글입니다.
        prompt = build_system_prompt(
            fitting(profile={"용도": "출근", "신경쓰는부위": ["shoulder_width"]}))
        block = prompt.split("[지난 대화에서 알게 된 것]")[1].split("[지금 보고 있는 옷]")[0]

        assert "신경 쓰는 부위: 어깨" in block
        assert "shoulder_width" not in block

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


class TestGarmentRules:
    """의류 파트 초안에서 온 규칙. 파일이 갈아치워져도 빠지면 안 되는 것들입니다."""

    def test_forbids_self_calculation(self):
        # AI 가 자체 계산하면 화면 리포트와 답변이 다른 값을 말할 수 있습니다.
        assert "수치를 직접 계산하지 마십시오" in build_system_prompt(fitting())

    def test_lists_judged_parts_per_garment(self):
        # 셔츠 상담에서 허리를 말하면 근거 수치가 없습니다.
        prompt = build_system_prompt(fitting())
        assert "셔츠 상담에서 허리를 언급하지 마십시오" in prompt

    def test_shoulder_uses_deviation_only(self):
        # 어깨는 정상 착용에도 actual_ease 가 -8cm 안팎입니다.
        assert "어깨는 어떤 경우에도" in build_system_prompt(fitting())

    def test_too_small_is_preview_wording(self):
        # 8/14 개정 — 3건 중 1건은 옷이 오히려 큽니다.
        prompt = build_system_prompt(fitting())
        assert "이 사이즈는 3D 미리보기가 없습니다" in prompt
        assert "옷이 작아 착용이 어렵습니다" not in prompt

    def test_overfit_multi_size_answer_exists(self):
        # shirt_over 는 12버킷 중 9개에서 두세 사이즈가 동시에 적정입니다.
        assert "아무거나 사도 되나요" in build_system_prompt(fitting())

    def test_forbids_list_markup_in_answer(self):
        # 음성으로 읽히므로 목록 기호를 읽을 수 없습니다.
        assert "목록·표·머리글 기호를 쓰지 마십시오" in build_system_prompt(fitting())


class TestNoGarmentSelected:
    """옷을 고르지 않은 상태.

    피팅룸에 들어왔지만 아직 옷을 안 고른 화면입니다. 이 상태에서 챗봇이
    **없는 옷을 설명하거나 지난 옷과 비교하던 문제**를 막습니다.

    원인은 프롬프트가 이행 불가능한 지시를 남겨 둔 것이었습니다. 머리말이
    "[지금 보고 있는 옷]" 인데 옷 정보가 없고, 지난 피팅 블록은 "지금 옷과
    비교하세요" 라고 했습니다. 지시를 못 지키게 되면 모델은 지시를 버리는
    대신 전제를 만들어 냅니다 — 지난 옷을 지금 옷처럼 말했습니다.
    """

    def _request(self, past=True):
        return ChatRequest(
            mode="fitting",
            message="이거 어때요?",
            fit_context=FitContext(
                measurements={"shoulder_width": 45.5, "chest_circ": 92.0},
                garment_id=None,
                profile={"용도": "출근"},
                past_fittings=[{"garment_id": "shirt_slim", "size": "m",
                                "wearable": False,
                                "tight_parts": ["shoulder_width"]}] if past else [],
            ),
        )

    def test_does_not_claim_a_garment_is_being_viewed(self):
        prompt = build_system_prompt(self._request())

        assert "[지금 보고 있는 옷]" not in prompt
        assert "[사용자 치수]" in prompt

    def test_tells_the_model_no_garment_is_chosen(self):
        prompt = build_system_prompt(self._request())

        assert "아직 옷을 고르지 않았습니다" in prompt
        assert "옷에 대한 판정을 말하지 마십시오" in prompt

    def test_does_not_ask_to_compare_with_a_garment_that_is_not_there(self):
        # "지금 옷과 비교하세요" 가 남아 있으면 모델이 비교 대상을 만들어 냅니다.
        prompt = build_system_prompt(self._request())

        assert "지금 옷과 수치 차이" not in prompt
        assert "지금 고른 옷이 없습니다" in prompt

    def test_still_sends_measurements(self):
        # 치수는 남아야 합니다. 사용자가 "제 어깨 몇이에요?" 를 물을 수 있습니다.
        prompt = build_system_prompt(self._request())

        assert "어깨 45.5cm" in prompt

    def test_garment_selected_keeps_the_original_wording(self):
        # 옷이 있을 때는 기존 동작이 그대로여야 합니다.
        request = ChatRequest(
            mode="fitting", message="이거 어때요?",
            fit_context=FitContext(
                measurements={"chest_circ": 92.0},
                garment_id="shirt_over", size="m", fit="오버핏",
                fit_report=[{"part": "chest_circ", "actual_ease": 32.0,
                             "ref_ease": 30.0, "deviation": 2.0, "verdict": "good"}],
                past_fittings=[{"garment_id": "shirt_slim", "size": "m",
                                "wearable": False, "tight_parts": []}],
            ),
        )

        prompt = build_system_prompt(request)

        assert "[지금 보고 있는 옷]" in prompt
        assert "지금 옷과 수치 차이가 1cm 이상일 때만 비교하세요" in prompt
        assert "아직 옷을 고르지 않았습니다" not in prompt

