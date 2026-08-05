"""1단계: 전신 사진 -> 관절 33개 + 실루엣 -> "키 대비 비율".

여기서 나오는 값은 전부 무차원 비율이다. cm 는 절대 계산하지 않는다.
사진은 두께 정보가 없으므로 둘레도 여기서 구하지 않는다.
절대 크기는 3단계(키 입력)에서, 둘레는 4단계(3D 메시)에서 정해진다.

설계 메모:
- 팔을 내린 사진에서는 실루엣에서 팔/손이 몸통에 붙어 한 덩어리가 된다.
  그대로 가로 스캔하면 가슴/허리/엉덩이 폭에 팔이 섞인다.
  -> 관절 골격으로 팔·손 영역을 지운 마스크에서 몸통 폭을 잰다.
- 측정 높이를 고정 비율로 박으면 사람마다 어긋난다.
  -> 목/허리는 "구간 내 최소폭", 가슴/엉덩이는 "구간 내 최대폭"으로 찾는다.
- 모든 결과는 사람 체형의 상식 범위와 대조해서, 벗어나면 None + 경고를 낸다.
  값이 조용히 틀린 채로 2단계에 넘어가는 것이 가장 위험하다.

실행:  ./venv/bin/python step1_photo.py [photo.jpg]
"""
import json
import sys

import cv2
import numpy as np
import mediapipe as mp

from . import config

# MediaPipe Pose 랜드마크 인덱스 (left/right 는 피사체 기준)
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# 몸통 폭을 재는 높이 (어깨 -> 골반 구간에서의 비율).
#
# 처음에는 "허리 = 가장 잘록한 곳" 을 탐색했지만 4/4 로 실패했다.
#   - 옷 입은 사진: 헐렁한 상의가 곧게 떨어져 잘록한 곳이 아예 없다
#   - 누드 메시 렌더: 정면 폭이 어깨에서 골반까지 단조 증가한다
#     (허리의 잘록함은 주로 깊이 방향이라 정면 실루엣에 거의 안 나타난다)
# 그래서 극값이 늘 탐색 구간 경계에서 잡혔다.
# 고정 비율로 재고, 사진 <-> 메시 정의 차이는 BIAS 가 흡수하게 한다.
TORSO_LEVELS = {"chest": 0.25, "waist": 0.70, "hip": 1.10}

# 키 대비 비율의 상식 범위. 벗어나면 측정이 오염된 것으로 본다.
# (성인 기준. 넉넉하게 잡되, 명백한 오측정은 걸러낼 수 있는 폭.)
SANE = {
    "shoulder_width":   (0.18, 0.26),
    "hip_width":        (0.09, 0.17),
    "torso_length":     (0.24, 0.34),
    "upper_arm_len":    (0.13, 0.21),
    "forearm_len":      (0.11, 0.19),
    "thigh_len":        (0.17, 0.27),
    "shank_len":        (0.15, 0.25),
    "leg_length":       (0.34, 0.50),
    "head_to_shoulder": (0.15, 0.26),
    "neck_width":       (0.030, 0.080),
    "shoulder_w_sil":   (0.19, 0.29),
    "chest_width":      (0.13, 0.24),
    "waist_width":      (0.10, 0.22),
    "hip_w_sil":        (0.13, 0.23),
    "thigh_width":      (0.055, 0.125),
    "calf_width":       (0.035, 0.085),
    "upper_arm_width":  (0.028, 0.075),
}


# ---------------------------------------------------------------- 검출

def load_pose(image_bgr):
    """MediaPipe Pose 실행. (랜드마크 픽셀좌표, 가시성, 세그멘테이션 마스크) 반환."""
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    ) as pose:
        res = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks is None:
        raise RuntimeError("사람을 찾지 못했습니다. 전신이 다 나온 사진인지 확인하세요.")

    h, w = image_bgr.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in res.pose_landmarks.landmark])
    vis = np.array([lm.visibility for lm in res.pose_landmarks.landmark])
    mask = (res.segmentation_mask > 0.5).astype(np.uint8)
    return pts, vis, mask, res.pose_landmarks


def clean_mask(mask):
    """가장 큰 연결 성분만 남겨 배경 노이즈를 제거한다."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise RuntimeError("실루엣이 비었습니다.")
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == biggest).astype(np.uint8)


def strip_arms(mask, pts, px_height):
    """골격을 따라 팔·손 영역을 지운 마스크와, 손을 지운 y 범위를 반환한다.

    팔을 내린 자세에서 몸통 폭을 재려면 필수다.
    어깨 관절 바로 아래부터 지워야 어깨선까지 깎아먹지 않는다.

    손을 지운 y 범위를 함께 돌려주는 이유: 팔을 내리면 손이 정확히 골반 높이에
    오는데, 그 높이에서는 손을 지우든 안 지우든 엉덩이 폭을 제대로 잴 수 없다.
    (지우면 옆구리가 패이고, 안 지우면 손 폭이 섞인다)
    호출부에서 이 범위와 겹치는 측정을 걸러내야 한다.
    """
    m = mask.copy()
    hand_ys = []
    arm_t = max(3, int(round(px_height * 0.075)))   # 팔뚝을 덮을 두께
    hand_r = max(4, int(round(px_height * 0.045)))  # 손 반경
    ipt = lambda p: (int(round(p[0])), int(round(p[1])))

    for sh, el, wr, pk, ix, th in (
        (L_SHOULDER, L_ELBOW, L_WRIST, L_PINKY, L_INDEX, L_THUMB),
        (R_SHOULDER, R_ELBOW, R_WRIST, R_PINKY, R_INDEX, R_THUMB),
    ):
        # 어깨에서 팔꿈치 쪽으로 25% 내려온 지점부터 시작 (어깨 보존)
        start = pts[sh] + 0.25 * (pts[el] - pts[sh])
        cv2.line(m, ipt(start), ipt(pts[el]), 0, arm_t)
        cv2.line(m, ipt(pts[el]), ipt(pts[wr]), 0, arm_t)
        for j in (wr, pk, ix, th):
            cv2.circle(m, ipt(pts[j]), hand_r, 0, -1)
            hand_ys += [pts[j][1] - hand_r, pts[j][1] + hand_r]

    return m, (float(min(hand_ys)), float(max(hand_ys)))


# ---------------------------------------------------------------- 폭 측정

def row_runs(mask, y, min_run=3):
    """y 행에서 몸이 차지하는 연속 구간들을 [(x0, x1), ...] 로 반환."""
    y = int(round(y))
    if not (0 <= y < mask.shape[0]):
        return []
    xs = np.flatnonzero(mask[y])
    if xs.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(xs) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [xs.size - 1]])
    runs = [(int(xs[s]), int(xs[e])) for s, e in zip(starts, ends)]
    return [r for r in runs if r[1] - r[0] >= min_run]


def run_at_x(mask, y, x, min_run=3):
    """y 행에서 x 를 품고 있는(또는 x 에 가장 가까운) 런의 폭."""
    runs = row_runs(mask, y, min_run)
    if not runs:
        return None
    for x0, x1 in runs:
        if x0 <= x <= x1:
            return float(x1 - x0)
    x0, x1 = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - x))
    return float(x1 - x0)


def widest_run(mask, y, min_run=3):
    """y 행에서 가장 넓은 런의 폭 (어깨처럼 팔까지 포함해야 하는 부위용)."""
    runs = row_runs(mask, y, min_run)
    if not runs:
        return None
    x0, x1 = max(runs, key=lambda r: r[1] - r[0])
    return float(x1 - x0)


def scan_extreme(mask, y_lo, y_hi, x_of_y, mode):
    """[y_lo, y_hi] 구간을 훑어 중심선 런의 폭이 최소/최대인 (y, width, at_edge) 를 찾는다.

    허리는 잘록한 곳, 가슴/엉덩이는 불룩한 곳이 해부학적 기준점이라
    고정 높이를 박는 것보다 이쪽이 사람마다 덜 어긋난다.

    at_edge=True 면 극값이 구간 경계에 붙었다는 뜻이고, 이는 진짜 극값이
    구간 밖에 있거나 실루엣이 깨졌다는 신호다. 값을 믿으면 안 된다.
    """
    y_lo, y_hi = int(round(y_lo)), int(round(y_hi))
    best = None
    for y in range(y_lo, y_hi + 1):
        w = run_at_x(mask, y, x_of_y(y))
        if w is None:
            continue
        if best is None or (w < best[1] if mode == "min" else w > best[1]):
            best = (float(y), w)
    if best is None:
        return None, None, False
    at_edge = (best[0] - y_lo) <= 2 or (y_hi - best[0]) <= 2
    return best[0], best[1], at_edge


def find_crotch(mask, hip_y, knee_y, cx, hold=8):
    """다리가 좌우로 갈라지는 y (가랑이). 허벅지는 반드시 이 아래에서 재야 한다.

    반드시 팔을 지운 마스크를 넣을 것. 원본 마스크에는 팔과 몸통 사이 틈이
    있어서, 그것을 가랑이로 오인한다.
    분리가 hold 행 연속으로 유지될 때만 인정해 노이즈를 거른다.
    """
    def split_at(y):
        runs = row_runs(mask, y, min_run=5)
        if len(runs) < 2:
            return False
        left = [r for r in runs if (r[0] + r[1]) / 2 < cx]
        right = [r for r in runs if (r[0] + r[1]) / 2 > cx]
        return bool(left and right)

    y_end = int(round(knee_y))
    for y in range(int(round(hip_y)), y_end):
        if all(split_at(y + k) for k in range(min(hold, y_end - y))):
            return float(y)
    return None


# ---------------------------------------------------------------- 본체

def analyze(photo_path, tag="", save_debug=True):
    """tag 를 주면 debug{tag}.jpg / silhouette{tag}.jpg 로 저장한다.
    캘리브레이션이 원본 산출물을 덮어쓰지 않게 하기 위한 것."""
    img = cv2.imread(str(photo_path))
    if img is None:
        raise FileNotFoundError(f"사진을 열 수 없습니다: {photo_path}")

    pts, vis, raw_mask, landmarks = load_pose(img)
    mask = clean_mask(raw_mask)

    mid = lambda a, b: (pts[a] + pts[b]) / 2.0
    shoulder_c, hip_c = mid(L_SHOULDER, R_SHOULDER), mid(L_HIP, R_HIP)
    knee_c, ankle_c = mid(L_KNEE, R_KNEE), mid(L_ANKLE, R_ANKLE)

    # --- 픽셀 기준 전신 높이: 실루엣의 최상단~최하단 ---
    # 랜드마크에는 정수리가 없어서 마스크로 잡는 편이 정확하다.
    ys = np.flatnonzero(mask.any(axis=1))
    top_y, bottom_y = float(ys[0]), float(ys[-1])
    px_height = bottom_y - top_y
    if px_height < 50:
        raise RuntimeError("실루엣이 너무 작습니다. 전신이 크게 나온 사진을 쓰세요.")

    torso = hip_c[1] - shoulder_c[1]
    no_arms, hand_band = strip_arms(mask, pts, px_height)

    # 몸통 중심선: 어깨 중심 -> 힙 중심을 잇는 직선으로 근사
    def center_x(y):
        t = (y - shoulder_c[1]) / torso if torso else 0.0
        return float(shoulder_c[0] + t * (hip_c[0] - shoulder_c[0]))

    warnings = []
    levels_px = {}   # 디버그 그림용: 이름 -> (y, width)

    def record(name, y, w, at_edge=False):
        levels_px[name] = (y, w)
        if y is None or w is None:
            warnings.append(f"{name}: 실루엣에서 폭을 찾지 못했습니다.")
            return None
        if at_edge:
            warnings.append(
                f"{name}: 극값이 탐색 구간 경계에서 잡혔습니다. "
                f"실루엣이 그 부근에서 깨졌을 가능성이 큽니다."
            )
            return None
        ratio = w / px_height
        lo, hi = SANE[name]
        if not (lo <= ratio <= hi):
            warnings.append(
                f"{name}: 비율 {ratio:.4f} 가 상식 범위 {lo}~{hi} 를 벗어났습니다. "
                f"실루엣이 오염됐을 가능성이 큽니다."
            )
            return None
        return round(ratio, 5)

    # --- 실루엣 기반 가로폭 ---
    widths = {}

    # 가랑이: 허벅지 측정의 상한선. 이 위에서 재면 두 다리를 합쳐서 재게 된다.
    crotch_y = find_crotch(no_arms, hip_c[1], knee_c[1], center_x(hip_c[1]))
    if crotch_y is None:
        warnings.append("가랑이를 찾지 못했습니다. 다리가 붙어 있는 자세로 보입니다.")

    # 목: 턱 아래 ~ 어깨선 사이에서 가장 좁은 곳
    jaw_y = float(max(pts[L_EAR][1], pts[R_EAR][1]))
    y, w, e = scan_extreme(no_arms, jaw_y, shoulder_c[1] - 0.02 * torso, center_x, "min")
    widths["neck_width"] = record("neck_width", y, w, e)

    # 어깨: 삼각근 포함이라 팔을 지우지 않은 마스크에서, 어깨선의 최대폭
    y = shoulder_c[1]
    w = widest_run(mask, y)
    widths["shoulder_w_sil"] = record("shoulder_w_sil", y, w)

    # 가슴 / 허리 / 엉덩이: 고정 비율 위치에서 몸통 폭. (TORSO_LEVELS 주석 참고)
    for name, key in (("chest_width", "chest"), ("waist_width", "waist"),
                      ("hip_w_sil", "hip")):
        y = shoulder_c[1] + TORSO_LEVELS[key] * torso
        # 손이 그 높이를 덮고 있으면 지워도 안 지워도 못 잰다. 아예 포기한다.
        if hand_band[0] <= y <= hand_band[1]:
            levels_px[name] = (y, None)
            warnings.append(
                f"{name}: 측정 높이에 손이 걸쳐 있습니다. "
                f"팔을 몸에서 벌린 사진이 필요합니다.")
            widths[name] = None
            continue
        widths[name] = record(name, y, run_at_x(no_arms, y, center_x(y)))

    # 허벅지: 반드시 가랑이 아래에서. 종아리: 무릎~발목 사이.
    # 좌우가 갈리므로 다리 골격 x 를 힌트로 준다.
    if crotch_y is not None:
        y = float(crotch_y + 0.08 * (knee_c[1] - crotch_y))
        t = (y - pts[L_HIP][1]) / (pts[L_KNEE][1] - pts[L_HIP][1])
        x = float(pts[L_HIP][0] + t * (pts[L_KNEE][0] - pts[L_HIP][0]))
        widths["thigh_width"] = record("thigh_width", y, run_at_x(no_arms, y, x, min_run=2))
    else:
        widths["thigh_width"] = record("thigh_width", None, None)

    y = float(pts[L_KNEE][1] + 0.30 * (pts[L_ANKLE][1] - pts[L_KNEE][1]))
    x = float(pts[L_KNEE][0] + 0.30 * (pts[L_ANKLE][0] - pts[L_KNEE][0]))
    widths["calf_width"] = record("calf_width", y, run_at_x(no_arms, y, x, min_run=2))

    # 상완: 팔을 지우지 않은 마스크에서, 어깨~팔꿈치 중간의 팔 쪽 런.
    # 팔이 몸통에 붙어 있으면 런이 하나로 합쳐져 몸통 폭을 팔 폭이라고 내놓게 된다.
    # 고른 런이 몸통 중심선을 품고 있으면 합쳐진 것이므로 포기한다.
    y = float((pts[L_SHOULDER][1] + pts[L_ELBOW][1]) / 2)
    x = float((pts[L_SHOULDER][0] + pts[L_ELBOW][0]) / 2)
    arm_runs = [r for r in row_runs(mask, y, 2) if r[0] <= x <= r[1]]
    if arm_runs and arm_runs[0][0] <= center_x(y) <= arm_runs[0][1]:
        levels_px["upper_arm_width"] = (y, None)
        warnings.append("upper_arm_width: 팔이 몸통에 붙어 분리되지 않습니다.")
        widths["upper_arm_width"] = None
    else:
        widths["upper_arm_width"] = record(
            "upper_arm_width", y, run_at_x(mask, y, x, min_run=2))

    # --- 관절 기반 길이 비율 ---
    dist = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))

    raw_lengths = {
        "shoulder_width":   dist(L_SHOULDER, R_SHOULDER),
        "hip_width":        dist(L_HIP, R_HIP),
        "torso_length":     abs(torso),
        "upper_arm_len":    (dist(L_SHOULDER, L_ELBOW) + dist(R_SHOULDER, R_ELBOW)) / 2,
        "forearm_len":      (dist(L_ELBOW, L_WRIST) + dist(R_ELBOW, R_WRIST)) / 2,
        "thigh_len":        (dist(L_HIP, L_KNEE) + dist(R_HIP, R_KNEE)) / 2,
        "shank_len":        (dist(L_KNEE, L_ANKLE) + dist(R_KNEE, R_ANKLE)) / 2,
        "leg_length":       abs(ankle_c[1] - hip_c[1]),
        "head_to_shoulder": abs(shoulder_c[1] - top_y),
    }
    lengths = {}
    for k, v in raw_lengths.items():
        ratio = v / px_height
        lo, hi = SANE[k]
        if lo <= ratio <= hi:
            lengths[k] = round(ratio, 5)
        else:
            lengths[k] = None
            warnings.append(f"{k}: 비율 {ratio:.4f} 가 상식 범위 {lo}~{hi} 를 벗어났습니다.")

    # --- 자세 / 촬영 품질 ---
    sh_tilt = float(abs(pts[L_SHOULDER][1] - pts[R_SHOULDER][1]) / px_height)
    hip_tilt = float(abs(pts[L_HIP][1] - pts[R_HIP][1]) / px_height)
    key_vis = vis[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE]]

    if sh_tilt > 0.02:
        warnings.append(f"어깨가 기울어져 있습니다 (tilt={sh_tilt:.3f}).")
    if hip_tilt > 0.02:
        warnings.append(f"골반이 기울어져 있습니다 (tilt={hip_tilt:.3f}).")
    if key_vis.min() < 0.6:
        warnings.append(f"주요 관절 가시성이 낮습니다 (min={key_vis.min():.2f}).")

    # 팔이 몸통에서 떨어져 있는지: 겨드랑이~손목 높이에서 런이 3개로 갈리는 비율
    y0, y1 = int(shoulder_c[1] + 0.15 * torso), int(pts[L_WRIST][1])
    split = [len(row_runs(mask, y)) >= 3 for y in range(y0, max(y0 + 1, y1))]
    arm_gap = float(np.mean(split)) if split else 0.0
    if arm_gap < 0.5:
        warnings.append(
            f"팔이 몸통에 붙어 있습니다 (분리 구간 {arm_gap:.0%}). "
            f"골격으로 팔을 지워 보정했지만, 팔을 30~45도 벌린 A자세 사진이 훨씬 정확합니다."
        )

    # --- 측정 높이를 정규화해서 함께 내보낸다 ---
    # 2단계가 "메시의 같은 위치"에서 재야 비교가 성립한다.
    # 고정 상수로 박아두면 사진마다 어긋난 곳을 비교하게 된다.
    def frac(name, a, b):
        y = levels_px.get(name, (None, None))[0]
        return None if y is None or a == b else round(float((y - a) / (b - a)), 4)

    levels = {
        "neck":  frac("neck_width", shoulder_c[1], top_y),      # 어깨->정수리
        "chest": frac("chest_width", shoulder_c[1], hip_c[1]),  # 어깨->골반
        "waist": frac("waist_width", shoulder_c[1], hip_c[1]),
        "hip":   frac("hip_w_sil", shoulder_c[1], hip_c[1]),
        "thigh": frac("thigh_width", hip_c[1], knee_c[1]),      # 골반->무릎
        "calf":  frac("calf_width", knee_c[1], ankle_c[1]),     # 무릎->발목
        "upper_arm": frac("upper_arm_width", pts[L_SHOULDER][1], pts[L_ELBOW][1]),
    }

    result = {
        "photo": str(photo_path),
        "px_height": round(px_height, 1),
        "image_size": [img.shape[1], img.shape[0]],
        "lengths": lengths,
        "widths": widths,
        "levels": levels,
        "quality": {
            "shoulder_tilt": round(sh_tilt, 4),
            "hip_tilt": round(hip_tilt, 4),
            "min_visibility": round(float(key_vis.min()), 3),
            "arm_separation": round(arm_gap, 3),
        },
        "warnings": warnings,
    }

    if save_debug:
        _save_debug(img, mask, no_arms, landmarks, levels_px, top_y, bottom_y, tag)
    return result


def _save_debug(img, mask, no_arms, landmarks, levels_px, top_y, bottom_y, tag=""):
    """debug.jpg (관절 + 측정선), silhouette.jpg (원본 | 팔 제거) 저장."""
    dbg = img.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        dbg, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )
    w_img = img.shape[1]
    for name, (y, width) in levels_px.items():
        if y is None:
            continue
        y = int(round(y))
        cv2.line(dbg, (0, y), (w_img, y), (0, 255, 255), 1)
        label = name + (f" {width:.0f}px" if width else " FAIL")
        cv2.putText(dbg, label, (5, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for y in (top_y, bottom_y):
        cv2.line(dbg, (0, int(y)), (w_img, int(y)), (255, 0, 255), 2)
    cv2.imwrite(str(config.OUT / f"debug{tag}.jpg"), dbg)

    # 왼쪽=원본 실루엣, 오른쪽=팔 제거 후 (몸통 폭을 재는 대상)
    left = np.zeros_like(img); left[mask > 0] = (255, 255, 255)
    right = np.zeros_like(img); right[no_arms > 0] = (120, 255, 120)
    cv2.imwrite(str(config.OUT / f"silhouette{tag}.jpg"), np.hstack([left, right]))


def main():
    photo = sys.argv[1] if len(sys.argv) > 1 else config.PHOTO
    res = analyze(photo)

    config.RATIOS_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"전신 픽셀 높이: {res['px_height']}px  (이미지 {res['image_size'][0]}x{res['image_size'][1]})")
    print("\n[길이 비율 = 부위 / 전신높이]")
    for k, v in res["lengths"].items():
        print(f"  {k:<18} {v if v is not None else '실패'}")
    print("\n[가로폭 비율 = 폭 / 전신높이]")
    for k, v in res["widths"].items():
        print(f"  {k:<18} {v if v is not None else '실패'}")
    print(f"\n자세: {res['quality']}")
    if res["warnings"]:
        print("\n⚠ 경고")
        for w in res["warnings"]:
            print(f"  - {w}")
    ok = sum(v is not None for v in {**res['lengths'], **res['widths']}.values())
    total = len(res['lengths']) + len(res['widths'])
    print(f"\n유효 항목: {ok}/{total}")
    print(f"저장: {config.RATIOS_JSON.name}, {config.DEBUG_JPG.name}, {config.SILHOUETTE_JPG.name}")


if __name__ == "__main__":
    main()
