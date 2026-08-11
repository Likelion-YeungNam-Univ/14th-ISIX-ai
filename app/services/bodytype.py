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

# 이 경고가 있으면 어깨 보조 문구를 아예 내보내지 않는다.
#
# 팔이 몸통에 붙은 사진에서는 어깨 실루엣이 팔까지 어깨로 잡아 과대 추정된다.
# 검증 사진이 정확히 그랬는데(165cm 에 45.6cm, 비율 0.276) 그대로 두면
# "어깨가 넓은 편입니다" 가 확언으로 사용자 화면에 나간다. 어깨를 판정에서
# 뺀 이유가 값을 믿을 수 없어서인데 문구는 확언으로 내보내면 모순이다.
#
# step1_photo.py 의 경고 문구와 짝이다. 문구가 바뀌면 이 가드가 조용히
# 풀리므로 함께 고쳐야 한다. 아래 세 곳에서 나온다.
#   step1_photo.py:304  "... 팔을 몸에서 벌린 사진이 필요합니다."
#   step1_photo.py:331  "upper_arm_width: 팔이 몸통에 붙어 분리되지 않습니다."
#   step1_photo.py:379  "팔이 몸통에 붙어 있습니다 (분리 구간 0%). ..."
ARM_WARNING_MARKERS = (
    "팔이 몸통에 붙어",
    "팔을 몸에서 벌린",
)

LABELS = {
    "hourglass": "모래시계형",
    "triangle": "삼각형",
    "inverted_triangle": "역삼각형",
    "rectangle": "직사각형",
    "round": "라운드형",
}

# 단정하지 않는 문구를 씁니다. 계측 오차가 ±4cm 이고 판정 경계가 5cm·18cm 라,
# 같은 사람이 다시 찍으면 타입이 바뀔 수 있습니다. "발달했습니다" 처럼 단정하면
# 그 변화가 사용자에게 모순으로 보입니다.
MESSAGES = {
    "hourglass": "가슴과 엉덩이가 비슷하고 허리가 뚜렷한 편입니다. "
                 "허리선이 있는 옷이 잘 맞습니다.",
    "triangle": "엉덩이가 상체보다 큰 편입니다. "
                "하의를 한 사이즈 크게 보시면 좋습니다.",
    "inverted_triangle": "어깨와 가슴이 하체보다 큰 편입니다. "
                         "상의 어깨 여유를 먼저 확인하세요.",
    "rectangle": "가슴·허리·엉덩이 차이가 크지 않은 편입니다. "
                 "직선적인 실루엣이 잘 맞습니다.",
    "round": "허리 둘레가 가슴·엉덩이보다 큰 편입니다. "
             "허리를 조이지 않는 상의가 편합니다.",
}

# 경계까지 거리가 이 안이면 두 타입을 함께 보여줍니다.
# 계측 오차 크기(±4cm)와 같게 두었습니다. 오차만큼 흔들리면 타입이 바뀌는
# 구간이라는 뜻이므로, 하나만 단정하지 않습니다.
#
# "경계에 걸쳐 있습니다" 같은 경고 문구를 따로 붙이는 방안도 검토했지만
# 넣지 않았습니다. 격자 12구간의 경계까지 거리가 중앙값 0.7cm(최소 0.2,
# 최대 3.9)로 대부분이 경계에 몰려 있어, 임계를 1cm 로 낮춰도 9/12 에
# 붙습니다. 항상 나오는 경고는 사용자가 읽지 않고 문구만 길어집니다.
# 두 타입을 함께 보여주는 것 자체가 이미 단정하지 않는 표현입니다.
SECONDARY_CM = 4.0

SHOULDER_NOTES = {
    "wide": "어깨가 넓은 편입니다. 상의는 어깨 치수를 먼저 확인하세요.",
    "narrow": "어깨가 좁은 편입니다. 오버핏 상의가 흘러내릴 수 있습니다.",
    "average": None,
}


def _has_arm_warning(warnings: Optional[list[str]]) -> bool:
    """어깨 계측을 믿을 수 없게 만드는 경고가 있는지."""
    if not warnings:
        return False
    return any(marker in w for w in warnings for marker in ARM_WARNING_MARKERS)


def _shoulder_note(shoulder_width: Optional[float],
                   height_cm: Optional[int],
                   warnings: Optional[list[str]]) -> Optional[str]:
    """어깨너비를 키로 정규화해 보조 문구를 만듭니다. 판정에는 쓰지 않습니다.

    팔 관련 경고가 있으면 문구를 만들지 않습니다. 계측이 과대 추정된 상태에서
    "어깨가 넓은 편입니다" 를 확언으로 내보내면 사용자를 잘못 안내합니다.
    """
    if _has_arm_warning(warnings):
        return None
    if not shoulder_width or not height_cm:
        return None
    ratio = shoulder_width / height_cm
    if ratio >= SHOULDER_WIDE_RATIO:
        return SHOULDER_NOTES["wide"]
    if ratio <= SHOULDER_NARROW_RATIO:
        return SHOULDER_NOTES["narrow"]
    return None


def _decide(chest: float, waist: float, hip: float) -> str:
    """세 둘레로 타입을 정합니다.

    판정 순서가 중요합니다. 허리가 가장 큰 경우를 먼저 걸러내지 않으면
    가슴·엉덩이 차만 보고 삼각형·역삼각형으로 잘못 분류됩니다.

    round 에 마진을 두는 이유 — 예전에는 waist >= max(chest, hip) 였습니다.
    그러면 가슴 85 / 엉덩이 85 에서 허리 84.9 는 rectangle, 85.0 은 round 가
    되어 0.1cm 에 결과가 뒤집힙니다. 계측 오차가 ±4cm 이므로 같은 사람이 다시
    찍으면 타입이 바뀝니다. 다른 경계는 모두 5cm 마진을 두고 있어 같은 기준으로
    통일했습니다.
    """
    if waist - max(chest, hip) >= BALANCED_CM:
        return "round"
    if abs(chest - hip) <= BALANCED_CM:
        return ("hourglass" if min(chest, hip) - waist >= WAIST_DEFINED_CM
                else "rectangle")
    return "triangle" if hip - chest > BALANCED_CM else "inverted_triangle"


def _nearest_alternative(chest: float, waist: float, hip: float,
                         body_type: str) -> tuple[Optional[str], float]:
    """가장 가까운 다른 타입과 그 경계까지의 거리(cm)를 돌려줍니다.

    임계값을 재설계하지 않고 표현만 완화하기 위한 계산입니다. 세 둘레를 각각
    조금씩 움직여 타입이 바뀌는 최소 거리를 찾습니다. 판정 규칙을 두 번 적는
    대신 _decide 를 다시 부르므로, 규칙을 고쳐도 이 함수는 따라옵니다.

    0.1cm 단위로 훑습니다. 계측이 소수 첫째 자리까지라 그보다 잘게 볼 의미가
    없고, 최대 6cm 까지만 봅니다(SECONDARY_CM 4cm 보다 넉넉하게).
    """
    best: tuple[Optional[str], float] = (None, float("inf"))
    for step in range(1, 61):
        delta = step / 10
        for chest_d, waist_d, hip_d in (
                (delta, 0, 0), (-delta, 0, 0),
                (0, delta, 0), (0, -delta, 0),
                (0, 0, delta), (0, 0, -delta)):
            other = _decide(chest + chest_d, waist + waist_d, hip + hip_d)
            if other != body_type:
                return other, delta
        if delta >= best[1]:
            break
    return best


def classify(measurements: dict[str, float],
             height_cm: Optional[int] = None,
             warnings: Optional[list[str]] = None) -> Optional[dict]:
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

    body_type = _decide(chest, waist, hip)
    alternative, distance = _nearest_alternative(chest, waist, hip, body_type)

    message = MESSAGES[body_type]

    # 경계에 가까우면 한 타입으로 단정하지 않습니다. 12구간 전부 ±4cm 오차에
    # 30% 이상 타입이 바뀌는 것을 확인했고, 임계값 재설계는 범위가 커서
    # 표현을 낮추는 쪽으로 정했습니다.
    if alternative is not None and distance <= SECONDARY_CM:
        message = f"{LABELS[alternative]}에 가까운 {LABELS[body_type]}입니다. {message}"
    # 역삼각형 문구에 이미 어깨 안내가 들어 있어 보조 문구를 덧붙이면 같은 말이
    # 두 번 나온다. 음성으로 읽어주면 특히 어색하다.
    if body_type != "inverted_triangle":
        note = _shoulder_note(measurements.get("shoulder_width"), height_cm,
                              warnings)
        if note:
            message = f"{message} {note}"

    return {
        "body_type": body_type,
        "body_type_label": LABELS[body_type],
        "body_type_message": message,
    }
