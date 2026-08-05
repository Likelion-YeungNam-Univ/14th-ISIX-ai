"""2단계: 사진 비율 -> SMPL-X beta (가중 최소제곱).

딥러닝 학습/파인튜닝 없음. scipy.optimize.least_squares 로 순수 수치 탐색이다.

실제 사용자 사진은 헐렁한 옷 / 팔 붙은 자세 / 신발이 기본이므로,
일부 비율이 None 이어도 최적화가 돌아가야 한다. 그래서:

  * 비율마다 신뢰도(weight)를 주고 가중 최소제곱으로 푼다.
    관절 기반은 실루엣과 무관해서 옷/자세에 안 흔들린다 -> 높은 가중치.
    실루엣 기반은 옷이 그대로 섞인다 -> 낮은 가중치.
    None 은 가중치 0 으로 자동 제외.
  * 제약이 모자라 언더컨스트레인트가 되면 세 가지로 막는다.
      (a) beta 정규화 (평균 체형에서 멀어지지 않게)
      (b) 몸무게 제약 (메시 부피 x 밀도 ~= 입력 몸무게)
      (c) 유효 제약 수에 따라 beta 차원 자체를 줄인다
  * 사용된 제약 수와 잔차로 confidence 를 뽑아 API 응답에 쓴다.

측정 정합성 메모:
  - 메시를 T포즈 그대로 재면 어깨 높이 x폭이 "양팔 span" 이 된다.
    고정 A자세로 포즈시킨 뒤, LBS 가중치로 팔 정점을 라벨링해 제외한다.
  - 측정 높이는 1단계가 내보낸 정규화 위치(levels)를 그대로 쓴다.
    상수로 박아두면 사진마다 다른 곳을 비교하게 된다.

실행:  ./venv/bin/python step2_shape.py --height 175 --weight 68 [--no-shoes]
"""
import argparse
import json

import numpy as np
import torch
from scipy.optimize import least_squares

from . import config

torch.set_grad_enabled(False)

# SMPL-X 본체 관절 인덱스
J_PELVIS = 0
J_L_HIP, J_R_HIP = 1, 2
J_L_KNEE, J_R_KNEE = 4, 5
J_L_ANKLE, J_R_ANKLE = 7, 8
J_NECK, J_HEAD = 12, 15
J_L_SH, J_R_SH = 16, 17
J_L_ELB, J_R_ELB = 18, 19
J_L_WRI, J_R_WRI = 20, 21

# LBS 지배 관절로 정점을 파트 분류할 때 쓰는 집합
L_ARM_JOINTS = {16, 18, 20} | set(range(25, 40))    # 왼손 관절 25~39
R_ARM_JOINTS = {17, 19, 21} | set(range(40, 55))    # 오른손 관절 40~54
ARM_JOINTS = L_ARM_JOINTS | R_ARM_JOINTS
L_LEG_JOINTS = {1, 4, 7, 10}
R_LEG_JOINTS = {2, 5, 8, 11}

# 팔을 내린 고정 A자세 (수직에서 25도 벌어짐)
ARM_DOWN_DEG = 65.0

# 야코비안 유한차분 스텝 (beta 단위). jac() 주석 참고.
JAC_STEP = 0.02


# ================================================================ 메시 계측

class Body:
    """SMPL-X 를 고정 A자세로 돌려, 1단계와 같은 정의의 비율을 뽑는다."""

    def __init__(self):
        import smplx
        self.model = smplx.create(
            str(config.SMPLX_DIR), model_type="smplx", gender=config.GENDER,
            num_betas=10, use_pca=False, ext="npz",
        )
        self.gender = config.GENDER
        self.faces = np.asarray(self.model.faces, dtype=np.int64)

        part = self.model.lbs_weights.detach().numpy().argmax(1)
        self.part = part
        self.is_arm = np.isin(part, sorted(ARM_JOINTS))
        self.is_larm = np.isin(part, sorted(L_ARM_JOINTS))
        self.is_rarm = np.isin(part, sorted(R_ARM_JOINTS))
        self.is_lleg = np.isin(part, sorted(L_LEG_JOINTS))
        self.is_rleg = np.isin(part, sorted(R_LEG_JOINTS))
        self.not_arm = ~self.is_arm

        # 어깨를 z축으로 회전시켜 팔을 내린다. body_pose 는 관절 1~21 이므로
        # 관절 16(L_shoulder) -> 인덱스 15, 관절 17(R_shoulder) -> 인덱스 16.
        bp = torch.zeros(1, 21 * 3)
        th = np.deg2rad(ARM_DOWN_DEG)
        bp[0, 15 * 3 + 2] = -th
        bp[0, 16 * 3 + 2] = +th
        self.body_pose = bp

    def forward(self, beta10):
        # torch 의 grad 모드는 스레드별(thread-local) 이다.
        # 모듈 최상단의 set_grad_enabled(False) 는 import 한 스레드에만 걸리므로,
        # 워커 스레드에서 호출하면 grad 가 켜진 채라 .numpy() 가 실패한다.
        # CLI(단일 스레드)에서는 안 나고 서버에서만 나는 버그라 여기서 직접 끈다.
        b = torch.tensor(np.asarray(beta10, dtype=np.float32)).reshape(1, 10)
        with torch.no_grad():
            out = self.model(betas=b, body_pose=self.body_pose, return_verts=True)
            return out.vertices[0].numpy(), out.joints[0].numpy()

    # ---------------------------------------------------------- 폭 도구

    def _x_extent(self, v, sel, y, h):
        """y 높이의 얇은 띠에서 x 방향 폭. 사진의 가로 스캔에 대응한다."""
        half = 0.006 * h
        for _ in range(5):
            m = sel & (np.abs(v[:, 1] - y) < half)
            if m.sum() >= 8:
                break
            half *= 1.8
        if m.sum() < 2:
            return None
        return float(v[m, 0].max() - v[m, 0].min())

    def _limb_perp(self, v, sel, A, B, t):
        """A->B 사지의 t 위치에서, 축에 수직인 폭.

        팔은 자세에 따라 기울어서 가로폭으로 재면 1/cos 만큼 부풀려진다.
        """
        d = (B - A)[:2]
        n = np.linalg.norm(d)
        if n < 1e-6:
            return None
        d = d / n
        p = np.array([-d[1], d[0]])
        P = (A + t * (B - A))[:2]
        rel = v[sel, :2] - P
        along, perp = rel @ d, rel @ p
        m = np.abs(along) < 0.10 * n
        if m.sum() < 4:
            return None
        return float(perp[m].max() - perp[m].min())

    # ---------------------------------------------------------- 본 계측

    def ratios(self, v, J, levels):
        """1단계와 같은 키 이름/정의로 무차원 비율을 낸다."""
        h = float(v[:, 1].max() - v[:, 1].min())
        top = float(v[:, 1].max())
        sh_c = (J[J_L_SH] + J[J_R_SH]) / 2
        hip_c = (J[J_L_HIP] + J[J_R_HIP]) / 2
        knee_c = (J[J_L_KNEE] + J[J_R_KNEE]) / 2
        ank_c = (J[J_L_ANKLE] + J[J_R_ANKLE]) / 2
        d = lambda a, b: float(np.linalg.norm(J[a] - J[b]))

        r = {
            "shoulder_width":   d(J_L_SH, J_R_SH) / h,
            "hip_width":        d(J_L_HIP, J_R_HIP) / h,
            "torso_length":     abs(sh_c[1] - hip_c[1]) / h,
            "upper_arm_len":    (d(J_L_SH, J_L_ELB) + d(J_R_SH, J_R_ELB)) / 2 / h,
            "forearm_len":      (d(J_L_ELB, J_L_WRI) + d(J_R_ELB, J_R_WRI)) / 2 / h,
            "thigh_len":        (d(J_L_HIP, J_L_KNEE) + d(J_R_HIP, J_R_KNEE)) / 2 / h,
            "shank_len":        (d(J_L_KNEE, J_L_ANKLE) + d(J_R_KNEE, J_R_ANKLE)) / 2 / h,
            "leg_length":       abs(ank_c[1] - hip_c[1]) / h,
            "head_to_shoulder": abs(top - sh_c[1]) / h,
        }

        torso = hip_c[1] - sh_c[1]          # 음수 (메시는 y 가 위쪽)
        leg = knee_c[1] - hip_c[1]          # 음수
        lv = lambda k, dflt: levels.get(k) if levels.get(k) is not None else dflt

        def px(name, y, sel):
            w = self._x_extent(v, sel, y, h)
            return None if w is None else w / h

        # 목 / 가슴 / 허리 / 엉덩이: 팔 정점 제외. 1단계가 준 위치에서 잰다.
        r["neck_width"] = px("neck", sh_c[1] + lv("neck", 0.35) * (top - sh_c[1]), self.not_arm)
        r["chest_width"] = px("chest", sh_c[1] + lv("chest", 0.25) * torso, self.not_arm)
        r["waist_width"] = px("waist", sh_c[1] + lv("waist", 0.68) * torso, self.not_arm)
        r["hip_w_sil"] = px("hip", sh_c[1] + lv("hip", 1.05) * torso, self.not_arm)

        # 어깨 실루엣 폭: 삼각근까지 포함해야 하므로 팔을 제외하지 않는다.
        # A자세라 어깨 높이에서는 팔이 아직 벌어지지 않아 삼각근 폭이 잡힌다.
        w = self._x_extent(v, np.ones(len(v), bool), sh_c[1], h)
        r["shoulder_w_sil"] = None if w is None else w / h

        # 허벅지 / 종아리: 한쪽 다리 정점만 써서 반대쪽이 섞이지 않게 한다.
        r["thigh_width"] = px("thigh", hip_c[1] + lv("thigh", 0.40) * leg, self.is_lleg)
        y_calf = knee_c[1] + lv("calf", 0.30) * (ank_c[1] - knee_c[1])
        r["calf_width"] = px("calf", y_calf, self.is_lleg)

        # 상완: 축에 수직으로 재야 자세 기울기에 안 흔들린다.
        w = self._limb_perp(v, self.is_arm, J[J_L_SH], J[J_L_ELB], lv("upper_arm", 0.5))
        r["upper_arm_width"] = None if w is None else w / h

        return r, h

    def volume(self, v):
        """부호 있는 사면체 합. 메시가 watertight 가 아니라 근사값이다."""
        t = v[self.faces]
        return float(abs(np.einsum("ij,ij->i",
                                   t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0)


# ================================================================ 최적화

def collect_targets(photo):
    """사진 비율을 하나로 합치고 (target, weight) 목록을 만든다. None 은 제외."""
    merged = {**photo["lengths"], **photo["widths"]}
    targets, weights, dropped = {}, {}, []
    for k, w in config.EFF_WEIGHTS.items():
        v = merged.get(k)
        if v is None:
            dropped.append(k)
            continue
        targets[k] = float(v) * config.BIAS.get(k, 1.0)
        weights[k] = w
    return targets, weights, dropped


def fit(photo, height_cm, weight_kg, use_shoe_fix=True, verbose=True, body=None,
        save_obj=True):
    body = body or Body()
    levels = photo.get("levels", {})

    targets, weights, dropped = collect_targets(photo)

    # 신발 보정: 사진의 전신 높이에는 밑창이 들어가 있는데 입력 키는 맨발 기준이다.
    # 모든 비율이 (키+밑창) 으로 나뉘어 있으므로 k = (키+밑창)/키 를 곱해 되돌린다.
    shoe_k = 1.0
    if use_shoe_fix:
        shoe_k = (height_cm + config.SHOE_CM) / height_cm
        targets = {k: v * shoe_k for k, v in targets.items()}

    keys = sorted(targets)
    n_valid = len(keys)
    n_beta = config.n_betas_for(n_valid)
    height_m = height_cm / 100.0

    w_arr = np.array([weights[k] for k in keys])
    t_arr = np.array([targets[k] for k in keys])

    def unpack(free):
        b = np.zeros(10)
        b[:n_beta] = free
        return b

    cache = {}

    def measured(free):
        key = tuple(np.round(free, 8))
        if key not in cache:
            if len(cache) > 4000:
                cache.clear()
            v, J = body.forward(unpack(free))
            r, mesh_h = body.ratios(v, J, levels)
            s = height_m / mesh_h                      # 입력 키로 스케일
            kg = body.volume(v) * s ** 3 * config.BODY_DENSITY
            cache[key] = (r, kg, mesh_h)
        return cache[key]

    def residuals(free):
        r, kg, mesh_h = measured(free)
        out = []
        for k, w, t in zip(keys, w_arr, t_arr):
            m = r.get(k)
            e = 0.0 if m is None else (m - t) / t        # 상대오차로 통일
            out.append(np.sqrt(w) * e)
        out.append(np.sqrt(config.W_WEIGHT) * (kg - weight_kg) / weight_kg)
        out.append(np.sqrt(config.W_HEIGHT) * (mesh_h - height_m) / height_m)
        out.extend(np.sqrt(config.REG) * free)          # beta 정규화
        return np.array(out)

    def jac(free):
        """야코비안을 절대 스텝으로 직접 계산한다. scipy 기본값에 맡기면 안 된다.

        이유 두 가지:
        1) least_squares 의 diff_step 은 abs_step = diff_step * |x0| 로 계산되는데
           x0=0 이면 0 이 되고, scipy 는 이를 감지해 기본값 1.5e-8 로 되돌린다.
           SMPL-X forward 는 float32 라 그 크기로는 정점이 전혀 안 움직인다.
           -> 야코비안이 통째로 0 이 되고 nfev=1 로 즉시 "수렴" 해버린다.
        2) 폭 측정이 정점 집합에 대한 max/min 이라 미세하게 계단형이다.
           스텝이 너무 작으면 계단 사이에 갇힌다.
        """
        f0 = residuals(free)
        out = np.empty((f0.size, free.size))
        for i in range(free.size):
            fp = free.copy()
            fp[i] += JAC_STEP
            out[:, i] = (residuals(fp) - f0) / JAC_STEP
        return out

    x0 = np.zeros(n_beta)
    B = config.BETA_BOUND
    sol = least_squares(residuals, x0, jac=jac, method="trf",
                        bounds=(-B, B), xtol=1e-10, ftol=1e-10, max_nfev=200)

    beta = unpack(sol.x)
    r_fin, kg_fin, mesh_h_fin = measured(sol.x)

    # --- 항목별 잔차 ---
    rows = []
    for k, w, t in zip(keys, w_arr, t_arr):
        m = r_fin.get(k)
        rows.append({
            "name": k, "weight": float(w), "target": round(float(t), 5),
            "mesh": None if m is None else round(float(m), 5),
            "rel_err": None if m is None else round(float((m - t) / t), 4),
        })

    errs = np.array([x["rel_err"] for x in rows if x["rel_err"] is not None])
    ws = np.array([x["weight"] for x in rows if x["rel_err"] is not None])
    rms = float(np.sqrt((ws * errs ** 2).sum() / ws.sum())) if len(errs) else 1.0

    # --- confidence ---
    coverage = float(ws.sum() / sum(config.EFF_WEIGHTS.values()))
    fit_score = float(1.0 / (1.0 + (rms / config.FIT_RMS_SCALE) ** 2))

    # beta 가 경계에 박혔다는 건 제약이 모자라 최적화가 폭주했다는 뜻이다.
    # 잔차만 보면 잘 맞은 것처럼 보이므로 반드시 따로 잡아내야 한다.
    saturated = [i for i, x in enumerate(sol.x)
                 if abs(abs(float(x)) - config.BETA_BOUND) < 1e-3]
    penalty = config.SATURATION_PENALTY if saturated else 1.0
    confidence = round(coverage * fit_score * penalty, 3)

    data_cost = float((residuals(sol.x)[:len(keys) + 2] ** 2).sum())
    reg_cost = float(config.REG * (sol.x ** 2).sum())

    result = {
        "beta": [round(float(x), 4) for x in beta],
        "n_betas_optimized": n_beta,
        "n_constraints": n_valid,
        "dropped": dropped,
        "height_cm": height_cm,
        "weight_kg_input": weight_kg,
        "weight_kg_estimated": round(kg_fin, 1),
        "intrinsic_height_cm": round(mesh_h_fin * 100, 1),
        "shoe_correction": round(shoe_k, 5) if use_shoe_fix else None,
        "residuals": rows,
        "weighted_rms_rel_err": round(rms, 4),
        "coverage": round(coverage, 3),
        "fit_score": round(fit_score, 3),
        "saturated_betas": saturated,
        "confidence": confidence,
        "cost_data": round(data_cost, 6),
        "cost_reg": round(reg_cost, 6),
        "optimizer": {"success": bool(sol.success), "nfev": int(sol.nfev),
                      "message": sol.message},
    }

    # --- 검증용 OBJ (입력 키로 스케일한 A자세 메시) ---
    # 서버에서는 끕니다. 로컬 검증용입니다.
    if save_obj:
        v, J = body.forward(beta)
        _, mesh_h = body.ratios(v, J, levels)
        v_scaled = (v - v.mean(0)) * (height_m / mesh_h)
        _save_obj(config.OUT / "body_est.obj", v_scaled, body.faces)

    if verbose:
        _report(result, dropped)
    return result


def _save_obj(path, verts, faces):
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces + 1:
            f.write(f"f {a} {b} {c}\n")


def _report(res, dropped):
    print(f"\nbeta({res['n_betas_optimized']}차원 최적화) =")
    print("  " + " ".join(f"{x:+.3f}" for x in res["beta"]))
    print(f"\n제약 {res['n_constraints']}개 사용, {len(dropped)}개 제외")
    if dropped:
        print(f"  제외: {', '.join(dropped)}")
    if res["shoe_correction"]:
        print(f"신발 보정 계수: x{res['shoe_correction']}  (밑창 {config.SHOE_CM}cm 가정)")

    print(f"\n{'항목':<18}{'w':>5}{'사진':>10}{'메시':>10}{'상대오차':>10}")
    for r in sorted(res["residuals"], key=lambda x: -abs(x["rel_err"] or 0)):
        e = "측정불가" if r["rel_err"] is None else f"{r['rel_err']:+.1%}"
        m = "-" if r["mesh"] is None else f"{r['mesh']:.4f}"
        print(f"  {r['name']:<16}{r['weight']:>5.1f}{r['target']:>10.4f}{m:>10}{e:>10}")

    print(f"\n몸무게: 입력 {res['weight_kg_input']}kg -> 메시 {res['weight_kg_estimated']}kg")
    print(f"비용: 데이터 {res['cost_data']:.5f} / 정규화 {res['cost_reg']:.5f}"
          f"  (정규화가 데이터를 압도하면 REG 를 낮출 것)")
    print(f"가중 RMS 상대오차: {res['weighted_rms_rel_err']:.2%}")
    sat = res["saturated_betas"]
    if sat:
        print(f"\n⚠ beta {sat} 가 경계(±{config.BETA_BOUND})에 박혔습니다. "
              f"제약이 모자라 최적화가 폭주한 상태이고, 잔차가 작아도 체형을 믿으면 안 됩니다.")
    print(f"\nconfidence = coverage {res['coverage']} x fit {res['fit_score']}"
          f"{f' x 포화패널티 {config.SATURATION_PENALTY}' if sat else ''} "
          f"= {res['confidence']}")
    print(f"수렴: {res['optimizer']['success']} (nfev={res['optimizer']['nfev']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, required=True, help="키 (cm, 맨발 기준)")
    ap.add_argument("--weight", type=float, required=True, help="몸무게 (kg)")
    ap.add_argument("--no-shoes", action="store_true", help="맨발 사진이면 지정")
    args = ap.parse_args()

    photo = json.loads(config.RATIOS_JSON.read_text())
    res = fit(photo, args.height, args.weight, use_shoe_fix=not args.no_shoes)

    config.BETA_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n저장: {config.BETA_JSON.name}, {config.BODY_OBJ.name}")
    print("→ body_est.obj 를 열어 체형이 사진과 닮았는지 반드시 눈으로 확인하세요.")


if __name__ == "__main__":
    main()
