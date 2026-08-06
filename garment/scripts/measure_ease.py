import json
from pathlib import Path

BUST = {'s': 85, 'm': 94, 'l': 100}
TARGET = {'shirt_slim': 8, 'tshirt_basic': 14, 'shirt_over': 30}
PANELS = ('left_ftorso', 'left_btorso', 'right_ftorso', 'right_btorso')

idx = json.loads(Path('patterns_index.json').read_text())
out = {}
for name, spec_path in idx.items():
    design, size = name.rsplit('_', 1)
    panels = json.loads(Path(spec_path).read_text())['pattern']['panels']
    tot = 0
    for pn in PANELS:
        if pn not in panels: break
        xs = [v[0] for v in panels[pn]['vertices']]
        tot += max(xs) - min(xs)
    else:
        ease = tot - BUST[size]
        t = TARGET.get(design)
        gap = f'   목표 {t:+d} → 차이 {ease-t:+6.1f}' if t else ''
        print(f'{name:20s} 옷둘레 {tot:6.1f}  여유 {ease:+6.1f}{gap}')
        out[name] = {'garment_cm': round(tot,1), 'ease_cm': round(ease,1)}

Path('ease_measured.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
