import numpy as np, trimesh
from scipy.spatial import cKDTree

K = 8          # 옷 정점 하나가 참조할 몸 정점 개수
SIGMA = 3.0    # 가중치 감쇠 폭 (cm)

def load_cm(path):
    m = trimesh.load(path, process=False)
    if m.vertices[:,1].max() < 10: m.apply_scale(100.0)
    return m

def warp(garment, src_body, dst_body):
    """src_body 위에서 시뮬한 옷을 dst_body 모양으로 변형"""
    assert len(src_body.vertices) == len(dst_body.vertices), '토폴로지 불일치'
    disp = dst_body.vertices - src_body.vertices          # 몸의 정점별 변위

    tree = cKDTree(src_body.vertices)
    dist, idx = tree.query(garment.vertices, k=K)         # 가장 가까운 몸 정점 K개
    w = np.exp(-(dist / SIGMA) ** 2)                      # 가까울수록 크게
    w /= w.sum(axis=1, keepdims=True)

    moved = garment.vertices + np.einsum('nk,nkj->nj', w, disp[idx])
    out = garment.copy()
    out.vertices = moved
    return out

def push_out(garment, body, margin=0.2):
    """몸 안으로 파고든 정점을 표면 밖으로 밀어냄"""
    tree = cKDTree(body.vertices)
    dist, idx = tree.query(garment.vertices)
    vn = body.vertex_normals[idx]
    outward = np.einsum('ij,ij->i', garment.vertices - body.vertices[idx], vn)
    inside = outward < margin
    fixed = garment.vertices.copy()
    fixed[inside] = body.vertices[idx][inside] + vn[inside] * margin
    print(f'  밀어낸 정점 {inside.sum()}개 / {len(fixed)}')
    garment.vertices = fixed
    return garment

if __name__ == '__main__':
    src = load_cm('assets/bodies/mean_female.obj')
    dst = load_cm('assets/bodies/mean_all.obj')
    g   = load_cm('Sim_results/tshirt_basic_m__mean_female.obj')

    print(f'몸 정점 {len(src.vertices)} / {len(dst.vertices)}')
    print(f'몸 변위 평균 {np.linalg.norm(dst.vertices-src.vertices,axis=1).mean():.2f}cm')

    w = warp(g, src, dst)
    w = push_out(w, dst)
    w.export('warp_test.glb')

    # 워핑 전후 몸까지 거리 비교
    for name, mesh in [('워핑 전', g), ('워핑 후', w)]:
        d, i = cKDTree(dst.vertices).query(mesh.vertices)
        vn = dst.vertex_normals[i]
        s = np.einsum('ij,ij->i', mesh.vertices - dst.vertices[i], vn)
        d = np.where(s < 0, -d, d)
        print(f'{name}: 평균 {d.mean():.2f}cm  관통 {(d<0).mean()*100:.1f}%')
