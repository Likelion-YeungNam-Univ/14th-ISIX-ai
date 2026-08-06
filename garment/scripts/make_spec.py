import json
from pathlib import Path

BODY = {'s': {'bust':85,'waist':69,'hips':91,'shoulder':37},
        'm': {'bust':94,'waist':78,'hips':100,'shoulder':39},
        'l': {'bust':100,'waist':84,'hips':106,'shoulder':40.5}}
FIT = {'shirt_slim':'슬림','tshirt_basic':'레귤러','shirt_over':'오버핏',
       'dress_basic':'레귤러','pants_slacks':'레귤러','skirt_pencil':'레귤러'}

def w(pn, keys):
    if any(k not in pn for k in keys): return None
    return round(sum(max(v[0] for v in pn[k]['vertices']) -
                     min(v[0] for v in pn[k]['vertices']) for k in keys), 1)

idx = json.loads(Path('patterns_index.json').read_text())
spec = {}
for name, path in idx.items():
    design, size = name.rsplit('_', 1)
    pn = json.loads(Path(path).read_text())['pattern']['panels']
    b = BODY[size]
    g, e = {}, {}

    bust = w(pn, ['left_ftorso','left_btorso','right_ftorso','right_btorso'])
    if bust: g['bust'], e['bust'] = bust, round(bust-b['bust'],1)

    waist = w(pn, ['wb_front','wb_back'])
    if waist: g['waist'], e['waist'] = waist, round(waist-b['waist'],1)

    hips = w(pn, ['skirt_front','skirt_back'])
    if hips: g['hips'], e['hips'] = hips, round(hips-b['hips'],1)

    if 'left_ftorso' in pn:
        vs = pn['left_ftorso']['vertices']
        ymax = max(v[1] for v in vs)
        tip = max((v for v in vs if v[1] > ymax*0.85), key=lambda v: v[0])
        sh = round(tip[0]*2, 1)
        g['shoulder'], e['shoulder'] = sh, round(sh-b['shoulder'],1)

    spec[name] = {'design': design, 'size': size.upper(), 'fit': FIT[design],
                  'garment_cm': g, 'ref_ease_cm': e}

Path('garment_spec.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False))
print(f'{len(spec)}개 저장 → garment_spec.json\n')
for n, v in spec.items():
    print(f"{n:20s} {v['fit']:5s} {v['ref_ease_cm']}")
