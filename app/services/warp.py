"""사전 시뮬한 옷을 사용자 아바타 모양으로 변형합니다.

**왜 필요한가.** 옷은 격자 대표 체형 12구간 위에서 미리 시뮬해 두고, 화면에는
사용자의 실제 아바타를 띄웁니다. 두 몸이 다른데 옷은 격자 몸 기준 좌표에
그대로 저장돼 있어서, 아바타의 어깨가 4cm 아래에 있으면 옷은 그 위 허공에
남고 몸통은 옷 표면을 뚫고 나옵니다.

실측(아바타 165cm · 격자 H2B0 170cm · 티셔츠 M):

    구간        격자 몸 위    아바타 위    파고든 정점
    어깨·목      +0.86cm      +2.56cm      0
    가슴         +1.69cm      +1.11cm      740
    허리         +4.25cm      +2.37cm      236
    합계         3개          1,178개 (14.6%)

어깨는 뜨는데 가슴은 파고듭니다. 반대 방향으로 동시에 어긋납니다.

**어떻게 고치나.** 옷을 다시 입히는 것이 아니라, 몸이 어떻게 변했는지를 옷에
전달합니다. 격자 몸과 아바타는 둘 다 SMPL-X 라 정점이 10,475개로 같고
정점 i 가 두 몸에서 같은 해부학적 지점입니다. 그래서 빼기만 하면 부위별
이동량이 나옵니다.

가중치가 거리에 따라 감쇠하므로(SIGMA) 몸에 붙은 부분만 몸을 따라가고 떠
있는 부분은 제자리에 남습니다. **옷이 몸을 따라 줄어들지 않습니다.**

    옷 상단     143.6 -> 139.5cm   (어깨에 걸림)
    몸과 간격   2.12 -> 2.63cm     (여유는 오히려 늘어남)

핏 순서도 유지됩니다 — 슬림 2.48 < 티셔츠 2.63 < 오버핏 2.95.

**정확도.** 같은 아바타에 실제 물리 시뮬을 돌려 비교했습니다.

    옷 상단      워핑 139.5 · 실제 시뮬 139.6   (1mm 차)
    정점 평균    워핑 0.81cm · 워핑 없음 1.26cm  (36% 개선)

물리가 아니라 기하 근사입니다. 격자 몸에서 만들어진 주름을 옮겨오는 것이라
새 체형에서 새로 생길 주름이나 중력으로 다시 흘러내리는 움직임은 담지
못합니다. 지금 체형 차이(키 5cm)에서는 그 효과가 0.81cm 수준입니다.

**비용.** 옷 한 벌에 0.014초. scipy KD-tree 와 numpy 연산뿐이라 GPU 가
필요 없습니다. 운영 서버에 GPU 가 없어 실제 시뮬(CPU 2~6분/벌)은 쓸 수
없지만 이 방식은 CPU 로 충분합니다.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from app.core.config import settings
from app.services import storage

logger = logging.getLogger(__name__)

# 워핑할 조합. 사전 시뮬이 있는 6종 x 3사이즈입니다.
DESIGNS = ("tshirt_basic", "shirt_slim", "shirt_over",
           "dress_basic", "pants_slacks", "skirt_pencil")
SIZES = ("s", "m", "l")

# 의류 GLB 가 올라가 있는 R2 접두어. 의류 파트가 관리합니다.
GARMENT_PREFIX = "garments/v1"

# 옷 정점 하나가 참조할 몸 정점 개수.
K = 8

# 가중치 감쇠 폭(cm). 이 값이 "옷이 몸을 따라 줄어들지 않는" 이유입니다.
# 크게 잡으면 멀리 떠 있는 정점까지 몸을 따라가 옷이 수축하고,
# 작게 잡으면 몸에 닿은 부분마저 안 따라가 파고듭니다.
SIGMA = 3.0

# 몸 표면에서 최소한 이만큼 띄웁니다(cm). 워핑만으로는 곡률이 심한 곳에서
# 소수의 정점이 남아 파고듭니다.
MARGIN = 0.2


def _to_cm(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """GLB 는 m, OBJ 는 cm 로 섞여 있어 cm 로 맞춥니다."""
    if mesh.vertices[:, 1].max() < 10:
        mesh.apply_scale(100.0)
    return mesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return _to_cm(m)


def warp(garment: trimesh.Trimesh,
         grid_body: trimesh.Trimesh,
         avatar: trimesh.Trimesh) -> trimesh.Trimesh:
    """격자 몸 위에서 시뮬한 옷을 아바타 모양으로 옮깁니다.

    두 몸의 토폴로지가 같아야 합니다. 다르면 정점 i 가 같은 지점을 가리키지
    않아 결과가 조용히 틀어지므로 여기서 막습니다.
    """
    if len(grid_body.vertices) != len(avatar.vertices):
        raise ValueError(
            "토폴로지 불일치: 격자 몸 %d, 아바타 %d. 둘 다 SMPL-X 여야 합니다."
            % (len(grid_body.vertices), len(avatar.vertices))
        )

    disp = avatar.vertices - grid_body.vertices           # 부위별 이동량
    dist, idx = cKDTree(grid_body.vertices).query(garment.vertices, k=K)
    w = np.exp(-(dist / SIGMA) ** 2)
    w /= w.sum(axis=1, keepdims=True)

    out = garment.copy()
    out.vertices = garment.vertices + np.einsum("nk,nkj->nj", w, disp[idx])
    return out


def push_out(garment: trimesh.Trimesh,
             avatar: trimesh.Trimesh,
             margin: float = MARGIN) -> tuple[trimesh.Trimesh, int]:
    """몸 안으로 파고든 정점을 표면 밖으로 밀어냅니다."""
    dist, idx = cKDTree(avatar.vertices).query(garment.vertices)
    vn = avatar.vertex_normals[idx]
    outward = np.einsum("ij,ij->i", garment.vertices - avatar.vertices[idx], vn)
    inside = outward < margin

    v = garment.vertices.copy()
    v[inside] = avatar.vertices[idx][inside] + vn[inside] * margin
    garment.vertices = v
    return garment, int(inside.sum())


def fit_to_avatar(garment_path: Path,
                  grid_body_path: Path,
                  avatar: trimesh.Trimesh,
                  out_path: Path) -> Optional[int]:
    """옷 한 벌을 아바타에 맞춰 저장합니다. 밀어낸 정점 수를 돌려줍니다.

    한 벌이 실패해도 나머지는 서빙해야 하므로 예외를 삼키고 None 을
    돌려줍니다. 호출부는 실패한 조합만 기존 경로로 폴백하면 됩니다.
    """
    try:
        garment = load_mesh(garment_path)
        grid = load_mesh(grid_body_path)
        fitted, pushed = push_out(warp(garment, grid, avatar), avatar)
        fitted.export(str(out_path))
        return pushed
    except Exception:
        logger.warning("워핑 실패: %s", garment_path.name, exc_info=True)
        return None


def build_grid_body(bucket: str, body=None) -> trimesh.Trimesh:
    """격자 대표 체형을 beta 로 다시 만듭니다.

    메시 파일을 배포에 싣지 않는 이유는 .gitignore 가 *.obj / *.glb 를 빼고
    있고 R2 에도 grid/ 접두어가 없기 때문입니다. body_grid.json 에 beta 와
    height_cm 이 이미 들어 있어 같은 함수로 다시 만들 수 있습니다.
    격자 몸이 원래 이 함수로 만들어졌습니다.

    body 를 넘기면 SMPL-X 모델(100MB) 재로딩을 건너뜁니다. 아바타 생성
    과정에서 이미 올려둔 것을 그대로 넘기십시오.

    ⚠ 재생성이 원본과 같은지는 verify_grid_rebuild 로 한 번 확인하십시오.
      1mm 만 어긋나도 워핑 결과가 오류 없이 조용히 틀어집니다.
    """
    from app.services.pipeline import step3_scale

    grid = json.loads(Path(settings.body_grid_path).read_text(encoding="utf-8"))
    spec = next(b for b in grid["buckets"] if b["id"] == bucket)
    mesh, _, _, _ = step3_scale.build(spec["beta"], spec["height_cm"], body=body)
    return _to_cm(mesh.copy())


def verify_grid_rebuild(bucket: str, reference_obj: Path, body=None) -> dict:
    """재생성한 격자 몸이 시뮬에 쓰인 원본과 같은지 확인합니다.

    운영에서는 쓰지 않습니다. 재생성 방식을 채택하기 전에 한 번 돌려
    max_cm 이 0 에 가까운지 보기 위한 것입니다. 0.1cm 를 넘으면 워핑
    기준이 어긋나므로 메시를 R2 에 올리는 쪽으로 가야 합니다.

        from app.services.warp import verify_grid_rebuild
        verify_grid_rebuild("H2B0", Path("assets/bodies/H2B0.obj"))
    """
    rebuilt = build_grid_body(bucket, body=body)
    original = load_mesh(reference_obj)
    if len(rebuilt.vertices) != len(original.vertices):
        return {"bucket": bucket, "ok": False,
                "reason": "정점 수 불일치 %d vs %d"
                          % (len(rebuilt.vertices), len(original.vertices))}
    d = np.linalg.norm(rebuilt.vertices - original.vertices, axis=1)
    return {"bucket": bucket, "ok": bool(d.max() < 0.1),
            "mean_cm": round(float(d.mean()), 4),
            "max_cm": round(float(d.max()), 4)}


def fit_all(avatar: trimesh.Trimesh, bucket: str, avatar_id: str,
            work_dir: Path, body=None) -> dict:
    """아바타에 맞춘 옷 18벌을 만들어 올리고 {조합: URL} 을 돌려줍니다.

    한 벌이 실패해도 나머지는 올립니다. 호출부와 백엔드는 결과에 없는
    조합만 기존 garments/v1/ 경로로 폴백하면 됩니다.

    의류 GLB 는 내용이 바뀌지 않으므로 work_dir 아래에 받아 두고 재사용합니다.
    """
    grid_mesh = build_grid_body(bucket, body=body)
    cache = work_dir / "garments"
    out_dir = work_dir / "fitted" / avatar_id
    out_dir.mkdir(parents=True, exist_ok=True)

    urls, failed = {}, []
    for design in DESIGNS:
        for size in SIZES:
            name = f"{design}_{size}__{bucket}.glb"
            src = cache / name
            if not src.exists() and not storage.download(f"{GARMENT_PREFIX}/{name}", src):
                failed.append(name)
                continue

            out = out_dir / f"{design}_{size}.glb"
            try:
                fitted, _ = push_out(warp(load_mesh(src), grid_mesh, avatar), avatar)
                fitted.export(str(out))
                url = storage.upload_to(
                    out, f"avatars/v1/{avatar_id}/{design}_{size}.glb")
                if url:
                    urls[f"{design}_{size}"] = url
            except Exception:
                logger.warning("워핑 실패: %s", name, exc_info=True)
                failed.append(name)

    if failed:
        logger.warning("[%s] 워핑 실패 %d건 — 기존 경로로 폴백됩니다: %s",
                       avatar_id, len(failed), ", ".join(failed[:5]))
    logger.info("[%s] 워핑 완료 %d/%d", avatar_id, len(urls),
                len(DESIGNS) * len(SIZES))
    return urls
