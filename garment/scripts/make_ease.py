import json, math, numpy as np, trimesh
from pathlib import Path
from scipy.spatial import cKDTree

OUT = Path('dist'); OUT.mkdir(exist_ok=True)
SPEC = json.loads(Path('garment_spec.json').read_text())
GRID = {b['id']: b['measurements_cm']
        for b in json.loads(Path('body_grid.json').read_text())['buckets']}
MISS = set(json.loads(Path('missing_combos.json').read_text())['missing'])

PART = {'bust': ('chest', 'chest_circ'),
        'waist': ('waist', 'waist_circ'),
        'hips': ('hip', 'hip_circ'),
        'shoulder': ('shoulder', 'shoulder_width')}

SCALE = {'unit': 'cm (둘레 여유)',
         'tight': {'max': 5.7, 'color': '#C82828'},
         'ok':    {'min': 5.7, 'max': 18.8, 'color': '#289656'},
         'loose': {'min': 18.8, 'color': '#285AC8'},
         'note': '반경 간격 0.9cm / 3.0cm 를 둘레로 환산한 값'}

bodies = {}
def body(name):
    if name not in bodies:
        m = trimesh.load(f'assets/bodies/{name}.obj', process=False)
        if m.vertices[:,1].max() < 10: m.apply_scale(100.0)
        bodies[name] = (m, cKDTree(m.vertices))
    return bodies[name]

n = 0
for obj in sorted(Path('Sim_results').glob('*.obj')):
    key, bname = obj.stem.split('__')
    name = f'{key}__{bname}'
    if key not in SPEC or name in MISS or bname not in GRID:
        continue

    g = trimesh.load(obj, process=False)
    b, tree = body(bname)

    dist, idx = tree.query(g.vertices)
    outward = np.einsum('ij,ij->i', g.vertices - b.vertices[idx], b.vertex_normals[idx])
    d = np.where(outward < 0, -dist, dist)

    nbr = g.vertex_neighbors                      # 중앙값 필터
    for _ in range(1):
        d = np.array([d[i] if not nbr[i] else np.median(np.append(d[nbr[i]], d[i]))
                      for i in range(len(d))])
    vertex_ease = np.round(d * 2 * math.pi, 1)    # 반경 → 둘레

    s, gm = SPEC[key], GRID[bname]
    parts = {}
    for gp, (out_key, grid_key) in PART.items():
        parts[out_key] = (round(s['garment_cm'][gp] - gm[grid_key], 1)
                          if gp in s['garment_cm'] and grid_key in gm else None)

    Path(OUT / f'{name}_ease.json').write_text(json.dumps({
        'garment_id': s['design'], 'size': s['size'], 'body_class': bname,
        'fit': s['fit'], 'parts': parts,
        'ref_ease': {PART[k][0]: v for k, v in s['ref_ease_cm'].items() if k in PART},
        'color_scale': SCALE,
        'vertex_count': len(vertex_ease),
        'vertex_ease': vertex_ease.tolist(),
    }, ensure_ascii=False))

    g.visual = trimesh.visual.ColorVisuals(mesh=g)   # 색 제거
    g.export(OUT / f'{name}.glb')
    n += 1

print(f'{n}쌍 생성 → dist/')
