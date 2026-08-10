"""계측 치수로 체형 유형을 판정합니다.

가슴·허리·엉덩이 **둘레** 세 개로 판정합니다.
요구사항명세서는 "어깨·허리·엉덩이 비율"이라고 적고 있지만, 어깨는
shoulder_width 로 너비(cm)이고 나머지는 둘레(cm)라 같은 축에서 비교할 수
없습니다. 너비를 둘레로 환산하려면 몸통 단면을 타원으로 가정해야 하고,
그 가정에서 나오는 오차가 판정 경계(5cm)보다 커서 타입이 뒤집힙니다.
그래서 어깨는 판정에서 빼고 키 대비 비율로 보조 표기만 합니다.

어깨를 뺀 또 다른 이유는 신뢰도입니다. 팔이 몸통에 붙은 사진에서는
실루엣이 팔까지 어깨로 잡아 shoulder_width 가 과대 추정됩니다. 실제로
검증 사진에서 165cm 에 어깨 45.6cm 가 나왔고 같은 요청의 warnings 에
"팔이 몸통에 붙어 있습니다" 가 함께 실렸습니다. 판정의 축으로 쓰기에는
불안정합니다.

경계값 근거 — 의류 사이즈 간격이 4cm(config.TOLERANCE_CM)이므로 그보다
작은 차이는 사이즈 선택을 바꾸지 못합니다. 가슴·엉덩이 차 5cm 를 유의미한
차이의 하한으로 둡니다.
"""

from typing import Optional

# 가슴·엉덩이 차가 이보다 작으면 "위아래가 비슷하다"로 본다.
# 계측 오차(±4cm)를 넘어서는 값이어야 판정이 뒤집히지 않는다.
BALANCED_CM = 5.0

# 허리가 가슴·엉덩이보다 이만큼 이상 작으면 허리가 뚜렷하다고 본다.
# 성인 여성 가슴-허리 차의 통상 범위 하단이다.
WAIST_DEFINED_CM = 18.0

# 어깨너비 / 키. 보조 표기용이며 판정에는 쓰지 않는다.
SHOULDER_WIDE_RATIO = 0.250
SHOULDER_NARROW_RATIO = 0.220

LABELS = {
    "hourglass": "모래시계형",
    "triangle": "삼각형",
    "inverted_triangle": "역삼각형",
    "rectangle": "직사각형",
    "round": "라운드형",
}

MESSAGES = {
    "hourglass": "가슴과 엉덩이가 비슷하고 허리가 뚜렷합니다. "
                 "허리선이 있는 옷이 잘 맞습니다.",
    "triangle": "엉덩이가 상체보다 발달했습니다. "
                "하의는 한 사이즈 크게, 상의는 몸에 맞게 고르는 편이 낫습니다.",
    "inverted_triangle": "어깨와 가슴이 하체보다 발달했습니다. "
                         "상의 어깨 여유를 먼저 확인하세요.",
    "rectangle": "가슴·허리·엉덩이 차이가 크지 않습니다. "
                 "직선적인 실루엣의 옷이 잘 맞습니다.",
    "round": "허리 둘레가 가슴·엉덩이보다 큽니다. "
             "허리를 조이지 않는 여유 있는 상의가 편합니다.",
}

SHOULDER_NOTES = {
    "wide": "어깨가 넓은 편입니다. 상의는 어깨 치수를 먼저 확인하세요.",
    "narrow": "어깨가 좁은 편입니다. 오버핏 상의가 흘러내릴 수 있습니다.",
    "average": None,
}


def _shoulder_note(shoulder_width: Optional[float],
                   height_cm: Optional[int]) -> Optional[str]:
    """어깨너비를 키로 정규화해 보조 문구를 만듭니다. 판정에는 쓰지 않습니다."""
    if not shoulder_width or not height_cm:
        return None
    ratio = shoulder_width / height_cm
    if ratio >= SHOULDER_WIDE_RATIO:
        return SHOULDER_NOTES["wide"]
    if ratio <= SHOULDER_NARROW_RATIO:
        return SHOULDER_NOTES["narrow"]
    return None


def classify(measurements: dict[str, float],
             height_cm: Optional[int] = None) -> Optional[dict]:
    """체형 유형을 판정합니다. 필요한 둘레가 없으면 None 을 반환합니다.

    None 을 반환하는 경우가 실제로 발생합니다. 계측이 실패한 부위는
    measurements 에서 아예 빠지므로(avatar_service 가 None 을 걸러냅니다)
    호출부는 체형 정보 없이도 아바타를 내려줄 수 있어야 합니다.
    """
    chest = measurements.get("chest_circ")
    waist = measurements.get("waist_circ")
    hip = measurements.get("hip_circ")
    if chest is None or waist is None or hip is None:
        return None

    # 판정 순서가 중요하다. 허리가 가장 큰 경우를 먼저 걸러내지 않으면
    # 가슴·엉덩이 차만 보고 삼각형·역삼각형으로 잘못 분류된다.
    if waist >= chest and waist >= hip:
        body_type = "round"
    elif abs(chest - hip) <= BALANCED_CM:
        smaller = min(chest, hip)
        body_type = ("hourglass" if smaller - waist >= WAIST_DEFINED_CM
                     else "rectangle")
    elif hip - chest > BALANCED_CM:
        body_type = "triangle"
    else:
        body_type = "inverted_triangle"

    message = MESSAGES[body_type]
    # 역삼각형 문구에 이미 어깨 안내가 들어 있어 보조 문구를 덧붙이면 같은 말이
    # 두 번 나온다. 음성으로 읽어주면 특히 어색하다.
    if body_type != "inverted_triangle":
        note = _shoulder_note(measurements.get("shoulder_width"), height_cm)
        if note:
            message = f"{message} {note}"

    return {
        "body_type": body_type,
        "body_type_label": LABELS[body_type],
        "body_type_message": message,
    }
