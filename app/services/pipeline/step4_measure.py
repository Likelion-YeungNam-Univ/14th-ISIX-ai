"""4단계: 3D 메시 단면에서 12부위 계측.

둘레는 반드시 여기서 잰다. 사진은 몸의 두께를 모르므로 픽셀에서 cm 를
계산할 수 없다. 3D 메시를 평면으로 잘라 그 단면의 둘레를 재는 것이
SMPL-X 를 거치는 이유다.

줄자와 단면 둘레의 차이:
  단면 외곽선을 그대로 따라가면 오목한 부분(가슴 사이, 척추 홈, 겨드랑이)까지
  파고든다. 실제 줄자는 팽팽히 당겨져 그런 곳을 가로지른다.
  -> 둘레는 단면의 볼록껍질(convex hull) 둘레로 잰다. 원시 둘레도 함께 남겨
     둘의 차이를 확인할 수 있게 한다.

측정 위치는 고정 상수가 아니라 해부학적 극값으로 찾는다.
  가슴/엉덩이/허벅지/팔 = 그 구간에서 가장 굵은 곳
  허리/목               = 그 구간에서 가장 가는 곳

실행:  ./venv/bin/python step4_measure.py
"""
import argparse
import json

import numpy as np
from scipy.spatial import ConvexHull

from . import config
from . import meshmeasure
from . import step2_shape as s2
from . import step3_scale

# 허벅지둘레 측정 높이: 가랑이에서 이만큼 아래 [m]
THIGH_BELOW_CROTCH = 0.03

# 정의가 애매해 팀 확인이 필요한 항목. 조용히 틀리면 사이즈 추천이 어긋난다.
NEEDS_CONFIRM = {
    "total_length": "목옆점(HPS) 바닥 높이로 정의했습니다. "
                    "옷의 총장이라면 밑단 위치를 팀에서 정해 빼야 합니다.",
    "front_width": "겨드랑이 높이에서 몸통 앞면의 좌우 직선 폭으로 정의했습니다.",
    "sleeve_length": "견봉(어깨 바깥위 모서리)에서 팔꿈치를 거쳐 손목뼈까지입니다. "
                     "VALIDATION_GUIDE.md 의 줄자 측정 정의와 동일합니다.",
    "back_length": "제7경추(C7, 고개 숙일 때 목뒤에 튀어나오는 뼈)에서 허리 "
                   "높이까지 등 표면을 따라간 길이입니다. "
                   "VALIDATION_GUIDE.md 의 줄자 측정 정의와 동일합니다.",
}


# ================================================================ 단면 도구
#
# 실제 계산은 meshmeasure.py 에 있다. 의류 시뮬(AI-2)이 같은 함수를 써야
# 여유량(옷 둘레 - 몸 둘레)이 성립하기 때문이다. 여기서는 이름만 끌어온다.

from .meshmeasure import (plane_basis, section_loops, perimeters,  # noqa: E402
                         contains as _contains, girth_at, girth_perp)


def nearest_loop(loops, point, normal=(0, 1, 0), min_perim=0.12):
    """meshmeasure.pick_loop 의 몸 계측용 래퍼 (바깥 루프 우선)."""
    return meshmeasure.pick_loop(loops, point, normal, min_perim, prefer="outer")


def scan_horizontal(mesh, y0, y1, near_xz, mode, n=30):
    """수평면을 훑어 볼록껍질 둘레가 최소/최대인 곳을 찾는다.

    극값이 구간 경계에서 잡히면 at_edge=True 로 표시한다. 진짜 극값이 구간
    밖에 있다는 뜻이고, 그 값은 해부학적 기준점이 아니다.
    """
    ys = np.linspace(y0, y1, n)
    best = None
    for y in ys:
        loops = section_loops(mesh, [0, y, 0], [0, 1, 0])
        L = nearest_loop(loops, np.array([near_xz[0], y, near_xz[1]]), [0, 1, 0])
        if L is None:
            continue
        raw, hull = perimeters(L, [0, 1, 0])
        if best is None or (hull < best["hull"] if mode == "min" else hull > best["hull"]):
            best = {"y": float(y), "raw": raw, "hull": hull, "loop": L,
                    "normal": np.array([0.0, 1, 0])}
    if best is not None:
        step = abs(ys[1] - ys[0])
        best["at_edge"] = (min(abs(best["y"] - ys[0]),
                               abs(best["y"] - ys[-1])) <= step * 1.01)
    return best


def find_hip(mesh, crotch_y, waist_y, n=30):
    """엉덩이둘레: 엉덩이가 가장 뒤로 튀어나온 높이에서 잰다.

    "구간 내 최대 둘레" 로 찾으면 안 된다. 가랑이 쪽으로 갈수록 단면이 양
    허벅지 윗부분을 함께 감싸서 둘레가 계속 커지고, 결국 가랑이(구간 경계)가
    잡힌다. 그건 줄자로 재는 엉덩이둘레가 아니다.
    ISO 8559 의 "둔부 최대 돌출부" 를 그대로 구현한다.
    """
    ys = np.linspace(crotch_y + 0.005, waist_y - 0.01, n)
    best = None
    for y in ys:
        L = nearest_loop(section_loops(mesh, [0, y, 0], [0, 1, 0]),
                         np.array([0.0, y, 0.0]), [0, 1, 0])
        if L is None:
            continue
        back = -float(L[:, 2].min())              # 뒤로 튀어나온 정도
        if best is None or back > best["back"]:
            raw, hull = perimeters(L, [0, 1, 0])
            best = {"y": float(y), "raw": raw, "hull": hull, "loop": L,
                    "back": back, "normal": np.array([0.0, 1, 0])}
    if best is not None:
        step = abs(ys[1] - ys[0])
        best["at_edge"] = (min(abs(best["y"] - ys[0]),
                               abs(best["y"] - ys[-1])) <= step * 1.01)
    return best


def scan_limb(mesh, A, B, t0, t1, mode, n=20):
    """사지 축에 수직인 평면으로 훑는다. 기울어진 팔을 수평으로 자르면 부풀려진다."""
    d = B - A
    d = d / np.linalg.norm(d)
    best = None
    for t in np.linspace(t0, t1, n):
        P = A + t * (B - A)
        L = nearest_loop(section_loops(mesh, P, d), P, d, min_perim=0.08)
        if L is None:
            continue
        raw, hull = perimeters(L, d)
        if best is None or (hull < best["hull"] if mode == "min" else hull > best["hull"]):
            best = {"t": float(t), "raw": raw, "hull": hull, "loop": L,
                    "y": float(P[1]), "normal": d, "point": P}
    if best is not None:
        step = abs(t1 - t0) / (n - 1)
        best["at_edge"] = (min(abs(best["t"] - t0),
                               abs(best["t"] - t1)) <= step * 1.01)
    return best


def submesh(mesh, vert_mask):
    """정점 마스크에 완전히 포함되는 면만 남긴 부분 메시.

    팔이 몸통에 붙어 있어서, 통짜 메시를 수평으로 자르면 몸통+양팔이
    한 폐곡선으로 나온다 (가슴둘레 119cm 처럼). LBS 라벨로 정확히 떼어낸다.
    잘린 자리에 구멍이 생기지만, 그 아래를 자르는 한 단면은 여전히 닫힌 곡선이다.
    """
    fm = vert_mask[mesh.faces].all(axis=1)
    if not fm.any():
        return None
    return mesh.submesh([np.flatnonzero(fm)], append=True)


# ================================================================ 해부학 지점

def find_crotch_y(mesh, y_lo, y_hi, n=80):
    """다리가 좌우로 갈라지는 가장 높은 y. 인심의 기준점이다.

    반드시 팔을 뗀 메시를 넣을 것. 통짜 메시에는 A자세에서 손이 골반 높이에
    오고, 손가락까지 세면 그 높이의 단면이 9조각으로 갈라진다. "좌우로 하나씩
    있으면 다리" 로 판정하면 손을 다리로 착각해 가랑이가 한참 위로 잡힌다.

    반환: (y, hit_top). hit_top 이면 탐색 상한까지 안 합쳐졌다는 뜻이고,
    그 값은 가랑이가 아니라 그냥 상한이다.
    """
    ys = np.linspace(y_lo, y_hi, n)
    best = None
    for y in ys:                                      # 아래에서 위로
        loops = section_loops(mesh, [0, y, 0], [0, 1, 0])
        left = [L for L in loops if L[:, 0].mean() < 0]
        right = [L for L in loops if L[:, 0].mean() > 0]
        if left and right:
            best = float(y)
        elif best is not None:
            return best, False                        # 합쳐졌으면 직전이 가랑이
    return best, (best is not None)


def find_c7(neck_section):
    """제7경추(C7, 목뒤점) 의 메시 표면 좌표. 등길이의 시작점이다.

    줄자로 등길이를 잴 때는 고개를 숙였을 때 목뒤에 튀어나오는 뼈에서 시작한다.
    사람이 짚을 수 있는 지점이어야 실측 비교가 성립하므로, 코드도 여기서 시작한다.
    (SMPL-X 목 관절 J12 는 몸 안쪽에 있어 줄자로 짚을 수 없다. 실제로 C7 보다
     5~7cm 아래여서, 그대로 쓰면 등길이가 구조적으로 짧게 나온다.)

    찾는 방법: "가장 튀어나온 점" 으로는 못 찾는다. SMPL-X 메시에는 C7 이
    혹으로 모델링되어 있지 않고, 등 표면 돌출은 어깨에서 위로 갈수록 단조
    감소하기만 한다. 그 규칙을 쓰면 어깨 높이가 잡힌다.
    대신 목 단면 둘레가 최소가 되는 높이(= 이미 neck_circ 를 재는 그 지점)의
    최후방 점을 쓴다.

    근거: 12구간 전체에서 이 높이 / 키 = 0.855~0.865 (흔들림 1%) 로 나왔고,
    문헌의 목뒤높이(경추점높이)/키 약 0.86 과 일치한다.
    """
    L = neck_section["loop"]
    return L[np.argmin(L[:, 2])]


def find_acromion(mesh, shoulder_y, side=+1):
    """어깨끝점(견봉). 소매길이의 시작점이다.

    재단에서 소매는 어깨의 바깥위 모서리 - 어깨선이 팔로 꺾이는 지점 - 에서
    시작한다. 어깨 관절 높이의 최외곽점을 쓰면 삼각근 한복판이라 4cm 쯤 아래고,
    소매길이가 구조적으로 짧게 나온다 (키 대비 0.300 vs 통상 0.32~0.35).

    어깨 영역에서 (x + y) 가 최대인 정점 = 바깥위 45도 방향의 모서리로 잡는다.
    12구간에서 소매/키 비율이 0.320~0.326 으로 안정적이었다.

    주의: shoulder_width 는 이 점을 쓰지 않는다. 어깨너비는 견봉이 아니라
    삼각근 바깥면 기준이 옷 어깨선과 더 잘 맞아 기존 정의를 유지한다.
    """
    v = mesh.vertices
    m = ((v[:, 1] > shoulder_y - 0.03) & (v[:, 1] < shoulder_y + 0.12)
         & (v[:, 0] * side > 0))
    if not m.any():
        return None
    sel = v[m]
    return sel[np.argmax(sel[:, 0] * side + sel[:, 1])]


def find_wrist_point(arm_mesh, elbow, wrist, side=+1):
    """손목뼈(척골 경상돌기) 근사. 소매길이의 끝점이다.

    줄자는 손목 관절 중심이 아니라 바깥으로 튀어나온 뼈에서 멈춘다.
    손목 높이 단면에서 팔 바깥쪽 표면점을 쓴다.
    (길이 기여는 0.1cm 수준이지만 정의를 맞춰 둔다.)
    """
    d = np.asarray(wrist, float) - np.asarray(elbow, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.asarray(wrist, float)
    d = d / n
    L = meshmeasure.pick_loop(section_loops(arm_mesh, wrist, d), wrist, d,
                              min_perim=0.05)
    if L is None:
        return np.asarray(wrist, float)
    lat = np.array([float(side), 0.0, 0.0])
    lat = lat - (lat @ d) * d
    nl = np.linalg.norm(lat)
    if nl < 1e-9:
        return np.asarray(wrist, float)
    return L[np.argmax(L @ (lat / nl))]


def find_armpit_y(mesh, is_arm):
    """겨드랑이 = 팔 정점과 몸통 정점이 함께 있는 면들 중 가장 낮은 y.

    "수평 단면이 3조각으로 갈라지는 높이" 로 찾으면 안 된다. 팔이 갈비뼈에
    붙어 있어서 실제 겨드랑이보다 한참 아래가 잡힌다.
    """
    f = mesh.faces
    boundary = is_arm[f].any(axis=1) & (~is_arm[f]).any(axis=1)
    if not boundary.any():
        return None
    return float(mesh.vertices[f[boundary]].reshape(-1, 3)[:, 1].min())


def back_path_length(mesh, y_top, y_bot, n=40):
    """등 중앙선을 따라간 표면 길이. 직선거리가 아니라 등의 곡률을 따라간다.

    정점을 직접 고르면 안 된다. SMPL-X 는 정중선에 정점 열이 없어서
    좁은 x 밴드에는 정점이 드문드문 있고, 어떤 높이에서는 앞면 정점만 걸린다.
    그러면 경로가 앞뒤로 번갈아 튀어 등길이가 키의 두 배로 나온다.
    높이마다 단면을 떠서 그 곡선의 최후방 점을 쓰면 항상 정의된다.

    팔을 뗀 몸통 메시를 넣으면 안 된다. 어깨에 뚫린 구멍 때문에 어깨~겨드랑이
    높이(전체 구간의 위쪽 1/3)에서 단면이 아예 안 잡히고, 경로가 아래쪽만
    덮어 등길이가 절반으로 나온다. 통짜 메시라도 최후방 점은 몸통 뒷면이므로
    팔이 섞여도 안전하다.
    """
    path = []
    for y in np.linspace(y_bot, y_top, n):
        L = nearest_loop(section_loops(mesh, [0, y, 0], [0, 1, 0]),
                         np.array([0.0, y, 0.0]), [0, 1, 0])
        if L is None:
            continue
        path.append([0.0, y, float(L[np.argmin(L[:, 2]), 2])])   # 등은 -z 쪽
    if len(path) < 2:
        return None, None
    path = np.array(path)
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), path


# ================================================================ 12부위 계측

def apply_calibration(meas):
    """메시 계측값 -> 줄자 기준값. step6_validate.py 가 만든 보정을 적용한다.

    기준점 정의 차이(예: SMPL-X 목 관절은 체내부, 재단은 등 표면의 C7 돌기)
    때문에 항목마다 계통 오차가 남는다. 실측 5명으로 그 편차를 잡는다.
    """
    calib = config.MEAS_CALIB
    if not calib:
        return dict(meas), False
    out = {}
    for k, v in meas.items():
        c = calib.get(k)
        if v is None or not c:
            out[k] = v
        elif c["mode"] == "offset":
            out[k] = round(v + c["value"], 1)
        else:
            out[k] = round(v * c["value"], 1)
    return out, True


def measure(mesh, J, height_cm, body, calibrate=True):
    v = mesh.vertices
    cm = lambda x: None if x is None else round(float(x) * 100, 1)

    # 팔을 뗀 몸통, 왼팔만, 왼다리만. 붙어 있는 부위끼리 섞이지 않게 한다.
    torso = submesh(mesh, body.not_arm)
    larm = submesh(mesh, body.is_larm)
    lleg = submesh(mesh, body.is_lleg)

    sh_c = (J[s2.J_L_SH] + J[s2.J_R_SH]) / 2
    hip_c = (J[s2.J_L_HIP] + J[s2.J_R_HIP]) / 2
    knee_c = (J[s2.J_L_KNEE] + J[s2.J_R_KNEE]) / 2
    tspan = hip_c[1] - sh_c[1]                        # 음수 (y 는 위쪽)

    out, marks = {}, {}

    # --- 둘레: 전부 팔을 뗀 몸통 / 한쪽 사지 메시에서 잰다 ---
    neck = scan_horizontal(torso, J[s2.J_NECK][1] + 0.01, J[s2.J_HEAD][1] - 0.02,
                           (0, 0), "min")
    chest = scan_horizontal(torso, sh_c[1] + 0.15 * tspan, sh_c[1] + 0.40 * tspan,
                            (0, 0), "max")
    # 허리 구간을 0.50 부터 잡으면 안 된다. 12구간 격자로 확인해보니 모든 체형에서
    # 최솟값이 0.50~0.57 부근에 몰려 있고, 비만 체형은 정확히 구간 위쪽 경계에서
    # 잡혔다. 진짜 허리는 그보다 위에 있다.
    waist = scan_horizontal(torso, sh_c[1] + 0.35 * tspan, sh_c[1] + 0.85 * tspan,
                            (0, 0), "min")

    # 가랑이는 팔을 뗀 메시에서, 고관절보다 충분히 위까지 훑는다.
    # SMPL-X 고관절(J1/J2)은 실제 가랑이보다 아래에 있어서 거기서 끊으면 못 찾는다.
    crotch_y, crotch_hit_top = find_crotch_y(
        torso, knee_c[1], hip_c[1] + 0.25 * (sh_c[1] - hip_c[1]))
    hip = find_hip(torso, crotch_y, waist["y"]) if (crotch_y and waist) else None

    # 허벅지둘레는 극값 탐색이 아니라 정해진 위치에서 잰다.
    # 표준 위치가 "가랑이 바로 아래 가장 굵은 곳" 이라, 구간 최대를 찾으면
    # 늘 구간 위쪽 끝이 잡힌다(=경계). 가랑이에서 THIGH_BELOW_CROTCH 만큼
    # 내려온 높이로 고정하는 편이 정의가 분명하다.
    # 다리 부분메시는 쓰지 않는다. LBS 라벨 경계가 들쭉날쭉해서 가랑이 근처에
    # 구멍이 생기고 단면이 안 잡힌다. 가랑이 아래는 통짜 메시에서도 두 다리가
    # 이미 분리되므로, 왼쪽 고관절 x 를 힌트로 왼다리 단면만 고르면 된다.
    thigh = None
    if crotch_y:
        y_th = crotch_y - THIGH_BELOW_CROTCH
        L = nearest_loop(section_loops(mesh, [0, y_th, 0], [0, 1, 0]),
                         np.array([float(J[s2.J_L_HIP][0]), y_th, 0.0]), [0, 1, 0])
        if L is not None:
            raw, hull = perimeters(L, [0, 1, 0])
            thigh = {"y": y_th, "raw": raw, "hull": hull, "loop": L,
                     "normal": np.array([0.0, 1, 0]), "at_edge": False}
    arm = scan_limb(larm, J[s2.J_L_SH], J[s2.J_L_ELB], 0.15, 0.55, "max")

    warnings = []
    if crotch_hit_top:
        warnings.append("가랑이를 못 찾았습니다 (탐색 상한까지 두 다리가 안 합쳐짐). "
                        "인심/허벅지/엉덩이가 전부 틀립니다.")
    for name, r in (("neck_circ", neck), ("chest_circ", chest), ("waist_circ", waist),
                    ("hip_circ", hip), ("thigh_circ", thigh), ("arm_circ", arm)):
        out[name] = cm(r["hull"]) if r else None
        if r:
            marks[name] = r
            if r.get("at_edge"):
                warnings.append(
                    f"{name}: 극값이 탐색 구간 경계에서 잡혔습니다 "
                    f"(높이 {r['y']*100:.1f}cm). 해부학적 기준점이 아닐 수 있습니다.")

    # --- 어깨너비: 삼각근 좌우 최대 폭 ---
    band = np.abs(v[:, 1] - sh_c[1]) < 0.02
    out["shoulder_width"] = cm(v[band, 0].max() - v[band, 0].min()) if band.any() else None
    if band.any():
        marks["shoulder_width"] = {"y": float(sh_c[1]),
                                   "x": (float(v[band, 0].min()), float(v[band, 0].max()))}

    # --- 앞품: 겨드랑이 높이에서 몸통 앞면 좌우 직선 폭 ---
    armpit_y = find_armpit_y(mesh, body.is_arm)
    if armpit_y is not None:
        loop = nearest_loop(section_loops(torso, [0, armpit_y, 0], [0, 1, 0]),
                            np.array([0.0, armpit_y, 0.0]), [0, 1, 0])
        if loop is not None:
            front = loop[loop[:, 2] > loop[:, 2].mean()]
            if len(front) >= 2:
                out["front_width"] = cm(front[:, 0].max() - front[:, 0].min())
                marks["front_width"] = {"y": armpit_y,
                                        "x": (float(front[:, 0].min()),
                                              float(front[:, 0].max()))}
    out.setdefault("front_width", None)

    # --- 등길이: C7(제7경추) -> 허리 높이, 등 표면을 따라 ---
    c7 = find_c7(neck) if neck else None
    if waist and c7 is not None:
        bl, bpath = back_path_length(mesh, float(c7[1]), waist["y"])
        out["back_length"] = cm(bl)
        if bpath is not None:
            marks["back_length"] = {"path": bpath, "c7": c7}
    else:
        out["back_length"] = None

    # --- 소매길이: 견봉 -> 팔꿈치 -> 손목뼈 ---
    # 표면을 따라가는 경로도 만들어 비교해봤지만, A자세에서 팔이 곧게 펴져 있어
    # 직선 체인과 차이가 0.7cm 이내였다 (오히려 팔꿈치 모서리를 깎아 더 짧았다).
    # 취약한 코드를 더 두지 않고 직선 체인을 유지하되, 양 끝점만 사람이 짚을 수
    # 있는 표면 랜드마크로 맞춘다.
    acromion = find_acromion(mesh, sh_c[1])
    if acromion is None:
        acromion = np.array([v[band, 0].max() if band.any() else J[s2.J_L_SH][0],
                             sh_c[1], J[s2.J_L_SH][2]])
    wrist_pt = (find_wrist_point(larm, J[s2.J_L_ELB], J[s2.J_L_WRI])
                if larm is not None else J[s2.J_L_WRI])
    chain = [acromion, J[s2.J_L_ELB], wrist_pt]
    out["sleeve_length"] = cm(sum(np.linalg.norm(b - a)
                                  for a, b in zip(chain, chain[1:])))
    marks["sleeve_length"] = {"chain": np.array(chain)}

    # --- 인심: 가랑이 높이 (발바닥 y=0 기준) ---
    out["inseam"] = cm(crotch_y)
    if crotch_y:
        marks["inseam"] = {"y": crotch_y}

    # --- 총장: 목옆점(HPS) 높이 ---
    # 목 원기둥 바깥이면서 목 최소단면보다 아래인 몸통 표면 중 가장 높은 점
    # = 어깨선이 목과 만나는 곳. 목 자체나 턱이 잡히지 않게 두 조건이 다 필요하다.
    hps_y = None
    if neck:
        nl = neck["loop"]
        cz = float(nl[:, 2].mean())
        r_neck = float(np.hypot(nl[:, 0], nl[:, 2] - cz).max())
        rad = np.hypot(v[:, 0], v[:, 2] - cz)
        ring = (rad > r_neck * 1.15) & (v[:, 1] < neck["y"]) & (v[:, 1] > sh_c[1] - 0.05)
        if ring.any():
            hps_y = float(v[ring, 1].max())
    out["total_length"] = cm(hps_y)
    if hps_y:
        marks["total_length"] = {"y": hps_y}

    # 보정 적용 여부는 항목별 경고가 아니라 실행 단위 정보다.
    # warnings 에 넣으면 격자 12구간 전부에 같은 문구가 붙어 진짜 경고를 덮는다.
    applied = False
    if calibrate:
        out, applied = apply_calibration(out)

    return (out, marks,
            {"crotch_y": crotch_y, "armpit_y": armpit_y,
             "c7_y": None if c7 is None else float(c7[1]),
             "calibrated": applied},
            warnings)


# ================================================================ 검증 그림

def save_check(mesh, out, marks, height_cm, path):
    import cv2

    v = np.asarray(mesh.vertices)
    H, W = 820, 420
    ppm = (H - 60) / (height_cm / 100 * 1.06)
    fx = lambda x: int(W / 2 + x * ppm)
    fz = lambda z: int(W / 2 - z * ppm)
    fy = lambda y: int(H - 30 - y * ppm)

    def body(proj):
        img = np.full((H, W, 3), 20, np.uint8)
        pts = np.array([[proj(a, c), fy(b)] for a, b, c in v])
        order = np.argsort(v[mesh.faces].mean(1)[:, 2] * (1 if proj is not fx else 1))
        for t in pts[mesh.faces][order]:
            cv2.fillConvexPoly(img, t.astype(np.int32), (86, 96, 112))
        return img

    front = body(lambda x, z: fx(x))
    side = body(lambda x, z: fz(z))

    Y = (0, 235, 255)
    for name in ("neck_circ", "chest_circ", "waist_circ", "hip_circ"):
        m = marks.get(name)
        if not m:
            continue
        y = fy(m["y"])
        cv2.line(front, (0, y), (W, y), Y, 1)
        cv2.line(side, (0, y), (W, y), Y, 1)
        cv2.putText(front, f"{name.replace('_circ','')} {out[name]}", (6, y - 4),
                    0, 0.45, Y, 1, cv2.LINE_AA)

    for name, col in (("shoulder_width", (120, 255, 120)),
                      ("front_width", (255, 180, 90))):
        m = marks.get(name)
        if not m:
            continue
        y = fy(m["y"])
        cv2.line(front, (fx(m["x"][0]), y), (fx(m["x"][1]), y), col, 2)
        cv2.putText(front, f"{name} {out[name]}", (6, y + 14), 0, 0.45, col, 1, cv2.LINE_AA)

    if "back_length" in marks:
        c7 = marks["back_length"].get("c7")
        if c7 is not None:
            cv2.circle(side, (fz(c7[2]), fy(c7[1])), 4, (255, 120, 255), -1)
            cv2.putText(side, "C7", (fz(c7[2]) + 7, fy(c7[1]) + 4), 0, 0.42,
                        (255, 120, 255), 1, cv2.LINE_AA)
        p = marks["back_length"]["path"]
        cv2.polylines(side, [np.array([[fz(c), fy(b)] for _, b, c in p], np.int32)],
                      False, (255, 120, 255), 2)
        cv2.putText(side, f"back_length {out['back_length']}", (6, 22), 0, 0.45,
                    (255, 120, 255), 1, cv2.LINE_AA)
    if "sleeve_length" in marks:
        c = marks["sleeve_length"]["chain"]
        cv2.polylines(front, [np.array([[fx(a), fy(b)] for a, b, _ in c], np.int32)],
                      False, (255, 255, 120), 2)
        cv2.putText(front, f"sleeve {out['sleeve_length']}", (W - 150, 22), 0, 0.45,
                    (255, 255, 120), 1, cv2.LINE_AA)
    for name, col in (("inseam", (140, 140, 255)), ("total_length", (200, 200, 200))):
        m = marks.get(name)
        if not m:
            continue
        y = fy(m["y"])
        cv2.line(front, (0, y), (W, y), col, 1)
        cv2.putText(front, f"{name} {out[name]}", (W - 170, y - 4), 0, 0.45, col, 1,
                    cv2.LINE_AA)

    for im, t in ((front, "FRONT"), (side, "SIDE")):
        cv2.putText(im, t, (W - 70, H - 8), 0, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.hstack([front, side]))


# ================================================================ 실행

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=None)
    args = ap.parse_args()

    est = json.loads(config.BETA_JSON.read_text())
    height_cm = args.height if args.height else est["height_cm"]
    mesh, J, info, body = step3_scale.build(est["beta"], height_cm)

    out, marks, extra, warnings = measure(mesh, J, height_cm, body)

    # 원시 둘레와의 차이 (줄자 근사가 얼마나 먹었는지)
    diff = {k: round((m["hull"] - m["raw"]) * 100, 2)
            for k, m in marks.items() if "hull" in m}

    result = {
        "height_cm": height_cm,
        "weight_kg": est["weight_kg_input"],
        "confidence": est.get("confidence"),
        "measurements_cm": {k: out[k] for k in config.MEASUREMENTS},
        "hull_minus_raw_cm": diff,
        "calibrated": extra["calibrated"],
        "landmarks_m": {k: (None if v is None else round(v, 4))
                        for k, v in extra.items() if k != "calibrated"},
        "warnings": warnings,
        "definitions_to_confirm": NEEDS_CONFIRM,
    }
    (config.OUT / "measurements.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))

    print(f"키 {height_cm}cm / 몸무게 {est['weight_kg_input']}kg / "
          f"confidence {est.get('confidence')}\n")
    print(f"{'부위':<18}{'값(cm)':>9}")
    missing = []
    for k in config.MEASUREMENTS:
        val = out[k]
        if val is None:
            missing.append(k)
        star = " *" if k in NEEDS_CONFIRM else ""
        print(f"  {k:<16}{'실패' if val is None else f'{val:>9.1f}'}{star}")

    print(f"\n볼록껍질 - 원시 둘레 [cm] (줄자가 오목부를 가로지른 양)")
    for k, d in diff.items():
        print(f"  {k:<16}{d:>+7.2f}")

    if not extra["calibrated"]:
        print("\n⚠ 실측 보정이 적용되지 않았습니다 (measure_calibration.json 없음).")
        print("  step6_validate.py 로 캘리브레이션하기 전 값은 계통 오차가 남아 있습니다.")
    print(f"\n가랑이 {extra['crotch_y']*100:.1f}cm / "
          f"겨드랑이 {extra['armpit_y']*100:.1f}cm (바닥 기준)")
    if missing:
        print(f"\n⚠ 측정 실패: {missing}")
    for w in warnings:
        print(f"⚠ {w}")
    print(f"\n* 표시 항목은 정의를 팀과 맞춰야 합니다:")
    for k, why in NEEDS_CONFIRM.items():
        print(f"  - {k}: {why}")

    check = config.OUT / "step4_check.jpg"
    save_check(mesh, out, marks, height_cm, check)
    print(f"\n저장: measurements.json, {check.name}")
    print("→ step4_check.jpg 에서 각 측정선이 실제 부위에 걸쳐 있는지 확인하세요.")


if __name__ == "__main__":
    main()
