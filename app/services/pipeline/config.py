"""파이프라인 공통 설정.

fittwin 로컬 파이프라인에서 가져왔습니다.
튜닝 상수(REG, W_HEIGHT 등)는 실측 스윕으로 정한 값이라 그대로 유지하고,
파일 경로만 서버 설정(app.core.config)에서 받도록 바꿨습니다.
"""
import json

from app.core.config import settings

# smplx.create() 는 smplx/ 의 부모 디렉터리를 받는다
SMPLX_DIR = settings.smplx_model_path.parent.parent

# 여성복 전용. 바꾸면 bias.json 과 body_grid.json 을 다시 만들어야 한다.
GENDER = "female"

# CLI 실행·디버그용 경로. 서버 요청 경로에서는 쓰지 않는다.
OUT = settings.avatar_dir.parent / "debug"
OUT.mkdir(parents=True, exist_ok=True)
PHOTO = OUT / "photo.jpg"
RATIOS_JSON = OUT / "photo_ratios.json"
DEBUG_JPG = OUT / "debug.jpg"
SILHOUETTE_JPG = OUT / "silhouette.jpg"

# --- 계측 12부위 (4단계에서 사용) ---
MEASUREMENTS = [
    "shoulder_width",   # 어깨너비
    "chest_circ",       # 가슴둘레
    "waist_circ",       # 허리둘레
    "hip_circ",         # 엉덩이둘레
    "neck_circ",        # 목둘레
    "arm_circ",         # 팔둘레(상완)
    "thigh_circ",       # 허벅지둘레
    "back_length",      # 등길이
    "sleeve_length",    # 소매길이
    "inseam",           # 인심
    "total_length",     # 총장
    "front_width",      # 앞품
]

# 의류 사이즈 간격 4cm -> 이보다 작아야 의미가 있다
TOLERANCE_CM = 4.0

# ---------------------------------------------------------------- 2단계


# ---------------------------------------------------------------- 3단계


# 스케일 계수가 이 범위를 벗어나면 경고한다.
# beta 최적화는 비율만 맞추므로 메시의 고유 높이는 입력 키와 무관하게 정해진다.
# 둘이 크게 어긋난다는 건 사진 비율이 이상하거나 키 입력이 틀렸다는 신호다.
SCALE_WARN = (0.90, 1.10)

# 신발 보정: 사진의 전신 높이에는 밑창이 포함되지만 입력 키는 맨발 기준이다.
# 보정 계수 k = (키 + 밑창) / 키 를 모든 비율에 곱해 맨발 기준으로 환산한다.
SHOE_CM = 2.5

# 인체 밀도 [kg/m^3]. 중립 메시(1.72m, 0.076m^3)에 곱하면 76.8kg 로 타당하지만,
# 메시가 watertight 가 아니라 부피에 약간의 오차가 있다.
# 7단계에서 5명 실측으로 반드시 재보정할 것.
BODY_DENSITY = 1010.0

# 비율별 신뢰도. 최적화의 가중 최소제곱에 그대로 들어간다.
#   관절 기반 = 실루엣과 무관해서 헐렁한 옷/붙은 팔에 영향을 안 받는다 -> 높게
#   실루엣 기반 = 옷/자세에 그대로 오염된다 -> 낮게
# None 인 항목은 자동으로 가중치 0 이 되어 빠진다.
W_JOINT = 1.0
W_SILHOUETTE = 0.3

WEIGHTS = {
    # 관절 기반 (MediaPipe 랜드마크 사이 거리)
    "shoulder_width":   W_JOINT,
    "hip_width":        W_JOINT,
    "torso_length":     W_JOINT,
    "upper_arm_len":    W_JOINT,
    "forearm_len":      W_JOINT,
    "thigh_len":        W_JOINT,
    "shank_len":        W_JOINT,
    "leg_length":       W_JOINT,
    "head_to_shoulder": W_JOINT * 0.5,   # 머리 모양/헤어스타일에 흔들린다
    # 실루엣 기반 (마스크 가로폭)
    "neck_width":       W_SILHOUETTE,
    "shoulder_w_sil":   W_SILHOUETTE,
    "chest_width":      W_SILHOUETTE,
    "waist_width":      W_SILHOUETTE,
    "hip_w_sil":        W_SILHOUETTE,
    "thigh_width":      W_SILHOUETTE,
    "calf_width":       W_SILHOUETTE,
    "upper_arm_width":  W_SILHOUETTE,
}

# 몸무게 제약의 가중치.
# 실루엣 폭이 전부 죽어도 "몸의 두께"를 알려주는 유일한 신호라 크게 준다.
W_WEIGHT = 3.0

# 키 제약의 가중치.
# 사진 비율은 스케일 불변이고, 몸무게 제약도 (부피 x (목표키/메시키)^3) 이라
# 사실상 스케일 불변이다. 즉 크기를 담당하는 beta[0] 은 목적함수에서 거의
# 자유 변수가 되어, 미세한 그래디언트에 끌려 엉뚱한 크기로 흘러간다.
# 그 상태로 3단계에서 균일 스케일하면 SMPL-X 가 학습한 "그 키의 체형"에서
# 벗어난 몸이 된다. 입력받은 키로 beta 의 고유 높이를 직접 묶어준다.
#
# 둘레는 사진에 안 보이는 "몸의 깊이" 가 결정하고, 그 깊이는 전적으로
# SMPL-X 의 사전분포에서 온다. 156cm 체형의 깊이 분포를 178cm 로 늘려 쓰면
# 그 사전분포가 어긋난다. 비율 RMS 를 조금 내주더라도 묶는 편이 낫다.
#
# 스윕 결과:
#   0    -> RMS 5.93%, 고유키 156.1cm, 3단계 스케일 x1.140  (폭주)
#   3.0  -> RMS 6.82%, 고유키 175.4cm, 스케일 x1.015        <- 채택
#   10.0 -> RMS 6.96%, 고유키 177.1cm, 스케일 x1.005
# 몸무게 추정은 어느 쪽이든 73.7kg 으로 동일했다 (스케일 불변이므로).
# 7단계에서 실측이 모이면 재검토할 것.
W_HEIGHT = 3.0

# confidence 의 잔차 항 척도. fit = 1 / (1 + (rms/scale)^2)
#   rms 5% -> 0.80,  10% -> 0.50,  20% -> 0.20
# 7단계에서 실측 오차와 대조해 재조정할 것.
FIT_RMS_SCALE = 0.10

# beta 정규화 세기. 클수록 평균 체형에 붙는다.
# 정면 측정만으로는 몸의 "두께" 방향이 제약되지 않아 최적화가 그 방향으로
# 무한정 도망간다. 그걸 막는 것이 이 항의 역할이다.
#
# 스윕 결과 (제약 15개, 10차원):
#   2e-4 -> RMS 3.93%  但 beta 4개 포화       (신뢰 불가)
#   1e-3 -> RMS 5.93%, max|b| 2.91, 포화 없음  <- 채택
#   5e-3 -> RMS 8.41%, max|b| 1.28            (과소적합 시작)
#   5e-2 -> RMS 9.47%, max|b| 0.31            (사실상 평균 체형)
REG = 1e-3

# beta 탐색 범위. SMPL-X beta 는 학습 데이터에서 표준화된 PCA 계수라
# 사실상 N(0,1) 스케일이다. |beta|>3 이면 이미 사람 체형을 벗어난다.
# 경계에 붙으면(포화) 제약이 모자라다는 신호이므로 confidence 를 깎는다.
BETA_BOUND = 3.0
SATURATION_PENALTY = 0.5

# 유효 제약 수에 따라 최적화할 beta 차원을 줄인다.
# SMPL-X beta 는 분산 순으로 정렬된 PCA 라 앞쪽 몇 개가 체형 변화의 대부분을 설명한다.
def n_betas_for(n_valid):
    if n_valid >= 12:
        return 10
    if n_valid >= 6:
        return 5
    return 3

# 사진 비율 -> 메시 비율 계통 오차 보정.
# MediaPipe 랜드마크와 SMPL-X 관절은 같은 점이 아니다.
# (예: MediaPipe 어깨 = 견봉, SMPL-X J16 = 어깨관절 중심 -> 사진 쪽이 더 넓게 나온다)
#
# calibrate_bias.py 가 만드는 bias.json 에서 읽는다. 하드코딩하지 않는다.
# 렌더 조건이나 측정 정의를 바꾸면 재캘리브레이션이 필요하므로,
# 값과 함께 생성 조건도 파일에 남긴다.
BIAS_JSON = settings.bias_path


# BIAS 가 체형에 따라 흔들린다는 건 상수 배율로는 못 잡는 정의 오차가 남아
# 있다는 뜻이다. 흔들린 만큼 그 항목의 신뢰도를 깎는다.
# spread == SPREAD_REF 이면 가중치가 절반이 된다.
SPREAD_REF = 0.10


def load_bias():
    """bias.json 에서 BIAS 와, 체형별 흔들림을 반영한 실효 가중치를 만든다."""
    if not BIAS_JSON.exists():
        return {k: 1.0 for k in WEIGHTS}, dict(WEIGHTS), None
    d = json.loads(BIAS_JSON.read_text())
    bias = {k: float(d["bias"].get(k, 1.0)) for k in WEIGHTS}
    spread = d.get("spread_across_shapes", {})
    eff = {}
    for k, w in WEIGHTS.items():
        s = spread.get(k)
        eff[k] = w if s is None else w / (1.0 + (float(s) / SPREAD_REF) ** 2)
    return bias, eff, d.get("render", None)


BIAS, EFF_WEIGHTS, BIAS_RENDER = load_bias()

# ---------------------------------------------------------------- 5단계

# 체형 12구간 격자. 의류 시뮬레이션/백엔드 파트가 그대로 가져다 쓴다.
# 하드코딩하지 않고 반드시 이 파일을 읽어서 쓸 것. 값이 어긋나면 엉뚱한 GLB 를
# 서빙하게 되고 에러 없이 조용히 틀린다.
GRID_JSON = settings.body_grid_path



# ---------------------------------------------------------------- 6단계



# 메시 계측 -> 줄자 기준 보정. step6_validate.py 가 만든다.
# 없으면 보정 없이 원값을 쓰고 경고한다.
MEAS_CALIB_JSON = settings.measure_calibration_path


def load_meas_calib():
    if not MEAS_CALIB_JSON.exists():
        return {}
    return json.loads(MEAS_CALIB_JSON.read_text()).get("calibration", {})


MEAS_CALIB = load_meas_calib()


def load_grid():
    if not GRID_JSON.exists():
        raise FileNotFoundError(
            f"{GRID_JSON.name} 이 없습니다. step5_grid.py 를 먼저 실행하세요.")
    return json.loads(GRID_JSON.read_text())

