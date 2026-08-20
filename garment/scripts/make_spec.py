import json
from pathlib import Path

# 기준 여유(ref_ease)를 빼는 몸입니다.
#
# 가슴/허리/엉덩이는 마네킹 치수입니다. 옷이 그 마네킹 위에서 제도됐고
# ref_ease 가 "이 옷이 의도한 여유" 이므로 기준도 같은 몸이어야 합니다.
#
# 어깨만 격자 몸(body_grid.json) 값입니다. 마네킹의 shoulder_w(37/39/40.5)는
# 제도용 치수라, make_ease.py 가 실제 여유를 계산할 때 쓰는 격자 몸의
# shoulder_width(42.9/46.0/47.6)와 정의가 달라 7cm 어긋납니다.
# 8/7 재보정 때 어깨만 격자로 옮겼는데 그 결정이 코드에 안 남아 있었습니다.
# 마네킹 값으로 두면 배포된 garment_spec.json(-7.9/-9.0/-8.9)을 재현하지
# 못하고, 이 스크립트를 다시 돌리는 순간 어깨 판정이 7cm 틀어집니다.
# 대응 버킷은 size_reference 의 mapped_bucket 을 따릅니다 (S→H1B0, M→H1B2, L→H1B3).
BODY = {'s': {'bust':85,'waist':69,'hips':91,'shoulder':42.9},
        'm': {'bust':94,'waist':78,'hips':100,'shoulder':46.0},
        'l': {'bust':100,'waist':84,'hips':106,'shoulder':47.6}}
FIT = {'shirt_slim':'슬림','tshirt_basic':'레귤러','shirt_over':'오버핏',
       'dress_basic':'레귤러','pants_slacks':'레귤러','skirt_pencil':'레귤러'}

def w(pn, keys):
    if any(k not in pn for k in keys): return None
    return round(sum(max(v[0] for v in pn[k]['vertices']) -
                     min(v[0] for v in pn[k]['vertices']) for k in keys), 1)

# patterns_index.json 은 {이름: {키: 경로}} 입니다. 키별로 패턴을 나눴지만
# 둘레는 키에 따라 변하지 않는 것을 36항목 전부에서 확인했으므로, 대표로
# 한 키만 읽습니다. 등급화하지 않은 품목은 '*' 한 벌뿐입니다.
CANON = 'H1'

idx = json.loads(Path('patterns_index.json').read_text())
spec = {}
for name, paths in idx.items():
    design, size = name.rsplit('_', 1)
    path = paths.get(CANON) or paths.get('*') or next(iter(paths.values()))
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
