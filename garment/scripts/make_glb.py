import json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial import cKDTree

OUT = Path('glb_out'); OUT.mkdir(exist_ok=True)
SPEC = json.loads(Path('garment_spec.json').read_text())
RED, GREEN, BLUE = np.array([200,40,40]), np.array([40,150,90]), np.array([40,90,200])

bodies = {}
def load_body(name):
    if name not in bodies:
        b = trimesh.load(f'assets/bodies/{name}.obj', process=False)
        if b.vertices[:,1].max() < 10: b.apply_scale(100.0)
        bodies[name] = (b, cKDTree(b.vertices), b.vertex_normals)
    return bodies[name]

def colors(d):
    """d: 부호 있는 간격(cm). 음수=관통"""
    c = np.tile(GREEN, (len(d), 1)).astype(float)
    tight = d < 0.9
    t = np.clip((0.9 - d[tight]) / 0.3, 0, 1)[:, None]
    c[tight] = GREEN + (RED - GREEN) * t
    loose = d > 3.0
    t = np.clip((d[loose] - 3.0) / 2.5, 0, 1)[:, None]
    c[loose] = GREEN + (BLUE - GREEN) * t
    return np.hstack([c.astype(np.uint8), np.full((len(d),1), 255, np.uint8)])

log = {}
for obj in sorted(Path('Sim_results').glob('*.obj')):
    key, body_name = obj.stem.split('__')
    if key not in SPEC: continue
    g = trimesh.load(obj, process=False)
    body, tree, vn = load_body(body_name)

    dist, idx = tree.query(g.vertices)
    outward = np.einsum('ij,ij->i', g.vertices - body.vertices[idx], vn[idx])
    d = np.where(outward < 0, -dist, dist)           # 몸 안이면 음수

    # 이웃 정점끼리 3회 평균 → 얼룩 제거
    nbr = g.vertex_neighbors
    for _ in range(2):
        d = np.array([d[i] if not nbr[i]
                      else np.median(np.append(d[nbr[i]], d[i]))
                      for i in range(len(d))])
    g.visual = trimesh.visual.ColorVisuals(mesh=g, vertex_colors=colors(d))
    name = f'{key}__{body_name}'
    g.export(OUT / f'{name}.glb')
    log[name] = {'fit': SPEC[key]['fit'],
                 'mean_gap': round(float(d.mean()), 2),
                 'pierce_pct': round(float((d < 0).mean()*100), 1),
                 'tight_pct': round(float((d < 0.9).mean()*100), 1),
                 'loose_pct': round(float((d > 3.0).mean()*100), 1),
                 'file': f'glb_out/{name}.glb'}

Path('heatmap_index.json').write_text(json.dumps(log, indent=2, ensure_ascii=False))
print(f'{len(log)}개 GLB 생성\n')
for k, v in log.items():
    if '__mean_female' not in k: continue
    print(f"{k.replace('__mean_female',''):20s} 평균간격 {v['mean_gap']:5.2f}cm  "
          f"관통 {v['pierce_pct']:5.1f}%  밀착 {v['tight_pct']:5.1f}%  여유 {v['loose_pct']:5.1f}%")
