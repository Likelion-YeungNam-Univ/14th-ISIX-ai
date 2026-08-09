"""3단계: beta -> 실제 크기의 메시 -> body.glb.

사진은 비율만 알려준다. 절대 크기는 입력받은 키가 결정한다.
2단계가 찾은 beta 는 "비율이 맞는 체형" 일 뿐이고, 그 메시의 고유 높이는
입력 키와 무관하게 정해진다. 여기서 전체를 균일 스케일해 키를 맞춘다.

균일 스케일이라 비율은 그대로 보존된다. 부피는 s^3 으로 커지는데,
2단계의 몸무게 제약도 같은 방식으로 스케일 후 부피를 썼으므로 일관된다.

내보내는 좌표계 (다른 파트가 그대로 가져다 쓴다):
  단위 m / Y 위 / +Z 정면 / 발바닥 y=0 / 좌우·앞뒤 원점 정렬
  포즈는 2단계 계측과 동일한 고정 A자세

실행:  ./venv/bin/python step3_scale.py
       ./venv/bin/python step3_scale.py --height 175
"""
import argparse
import json

import numpy as np
import trimesh

from . import config
from . import step2_shape


def build(beta, height_cm, body=None):
    """beta -> 입력 키에 맞춘 trimesh. (mesh, joints, 진단정보, body) 반환.

    관절도 정점과 똑같은 변환을 거쳐 나온다. 4단계가 이 좌표계에서 계측한다.
    body 를 넘기면 SMPL-X 모델 로딩(100MB)을 건너뛴다. 격자 12개를 돌릴 때 필요.
    """
    body = body or step2_shape.Body()
    v, J = body.forward(np.asarray(beta, dtype=np.float64))
    J = J.astype(np.float64)

    raw_h = float(v[:, 1].max() - v[:, 1].min())     # beta 가 만든 고유 높이
    target_h = height_cm / 100.0
    s = target_h / raw_h

    v, J = v * s, J * s
    # 발바닥을 y=0 에, 좌우/앞뒤는 원점에 맞춘다. 관절에도 같은 이동을 적용한다.
    off = np.array([(v[:, 0].min() + v[:, 0].max()) / 2,
                    v[:, 1].min(),
                    (v[:, 2].min() + v[:, 2].max()) / 2])
    v -= off
    J -= off

    mesh = trimesh.Trimesh(v, body.faces, process=False)
    info = {
        "intrinsic_height_m": round(raw_h, 4),
        "target_height_m": round(target_h, 4),
        "scale": round(s, 5),
        "volume_m3": round(float(abs(mesh.volume)), 6),
    }
    return mesh, J, info, body


def _save_check(mesh, height_cm, path):
    """cm 눈금 위에 메시를 올린 검증 그림. 접지(y=0)와 키를 눈으로 확인한다.

    정사영으로 그린다. 여기서는 사진 재현이 아니라 치수 확인이 목적이므로
    원근이 들어가면 오히려 눈금과 안 맞는다.
    """
    import cv2

    v = np.asarray(mesh.vertices)
    H, W = 760, 460
    top_cm = np.ceil(height_cm / 10) * 10 + 10          # 눈금 상한
    ppc = (H - 40) / top_cm                              # 픽셀/cm
    px = lambda x, y: (int(W / 2 + x * 100 * ppc), int(H - 20 - y * 100 * ppc))

    img = np.full((H, W, 3), 18, np.uint8)
    for cm in range(0, int(top_cm) + 1, 10):
        y = int(H - 20 - cm * ppc)
        major = cm % 50 == 0
        cv2.line(img, (0, y), (W, y), (90, 90, 90) if major else (45, 45, 45), 1)
        if major:
            cv2.putText(img, f"{cm}", (4, y - 3), 0, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    order = np.argsort(v[mesh.faces].mean(1)[:, 2])
    pts = np.array([px(a, b) for a, b in v[:, :2]])
    for t in pts[mesh.faces][order]:
        cv2.fillConvexPoly(img, t.astype(np.int32), (190, 205, 235))

    y_top, y_bot = px(0, v[:, 1].max())[1], px(0, v[:, 1].min())[1]
    for y, c in ((y_top, (80, 220, 80)), (y_bot, (80, 220, 80))):
        cv2.line(img, (0, y), (W, y), c, 1)
    cv2.putText(img, f"{height_cm:.1f}cm", (W - 118, y_top + 22), 0, 0.62,
                (80, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(img, "sole y=0", (W - 118, y_bot - 8), 0, 0.45,
                (80, 220, 80), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=None,
                    help="키 [cm]. 생략하면 step2_beta.json 값을 쓴다")
    args = ap.parse_args()

    est = json.loads(config.BETA_JSON.read_text())
    beta = est["beta"]
    height_cm = args.height if args.height else est["height_cm"]
    weight_kg = est["weight_kg_input"]

    mesh, _joints, info, _body = build(beta, height_cm)

    # --- 내보내기 ---
    mesh.export(str(config.BODY_GLB))

    # --- 검증: 다시 읽어서 키가 보존됐는지 확인 ---
    back = trimesh.load(str(config.BODY_GLB), force="mesh")
    bv = np.asarray(back.vertices)
    got_h = float(bv[:, 1].max() - bv[:, 1].min()) * 100
    err = got_h - height_cm

    density_kg = info["volume_m3"] * config.BODY_DENSITY

    meta = {
        "source": {
            "beta": beta,
            "confidence": est.get("confidence"),
            "n_constraints": est.get("n_constraints"),
        },
        "input": {"height_cm": height_cm, "weight_kg": weight_kg},
        "scaling": info,
        "coordinate_frame": {
            "units": "meters",
            "up": "+Y",
            "facing": "+Z",
            "origin": "발바닥 y=0, 좌우/앞뒤 중앙",
        },
        "pose": {
            "type": "A-pose (고정)",
            "arm_pose_angle_deg": step2_shape.ARM_POSE_ANGLE_DEG,
            "forearm_angle_deg": step2_shape.FOREARM_ANGLE_DEG,
            "angle_reference": "수직 아래 = 0도",
            "note": "2단계 계측과 동일한 포즈. 의류 시뮬레이션은 이 포즈를 기준으로 할 것.",
        },
        "topology": {
            "model": "SMPL-X neutral",
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "note": "SMPL-X 표준 정점 순서. 다른 beta 로 만들어도 순서는 동일하다.",
        },
        "verification": {
            "reloaded_height_cm": round(got_h, 3),
            "height_error_cm": round(err, 4),
            "implied_weight_kg": round(density_kg, 1),
        },
    }
    config.BODY_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # --- 보고 ---
    print(f"beta 고유 높이  {info['intrinsic_height_m']*100:.1f}cm")
    print(f"입력 키         {height_cm:.1f}cm")
    print(f"스케일 계수     x{info['scale']:.4f}")
    print(f"\n부피 {info['volume_m3']:.5f} m3 -> 몸무게 환산 {density_kg:.1f}kg "
          f"(입력 {weight_kg}kg)")
    print(f"\n검증: GLB 재로드 높이 {got_h:.2f}cm  (오차 {err:+.4f}cm)")
    print(f"      정점 {len(mesh.vertices)}  면 {len(mesh.faces)}  "
          f"watertight={mesh.is_watertight}")

    lo, hi = config.SCALE_WARN
    if not (lo <= info["scale"] <= hi):
        print(f"\n⚠ 스케일 계수가 {lo}~{hi} 범위를 벗어났습니다.")
        print(f"  beta 가 만든 체형은 {info['intrinsic_height_m']*100:.0f}cm 인데 "
              f"{height_cm:.0f}cm 로 늘리고 있습니다.")
        print(f"  사진 비율이 이상하거나, 입력 키가 실제와 다를 수 있습니다.")
    if abs(density_kg - weight_kg) > 5:
        print(f"\n⚠ 부피에서 환산한 몸무게가 입력과 {abs(density_kg-weight_kg):.1f}kg "
              f"차이납니다. BODY_DENSITY 재보정이 필요할 수 있습니다.")

    check = config.OUT / "step3_check.jpg"
    _save_check(mesh, height_cm, check)
    print(f"\n저장: {config.BODY_GLB.name}, {config.BODY_META.name}, {check.name}")
    print("→ step3_check.jpg 에서 발바닥이 0cm 선에, 정수리가 키 선에 닿는지 확인하세요.")


if __name__ == "__main__":
    main()
