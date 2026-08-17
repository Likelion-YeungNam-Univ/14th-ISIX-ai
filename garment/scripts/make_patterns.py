import yaml, json
from pathlib import Path
from assets.garment_programs.meta_garment import MetaGarment
from assets.bodies.body_params import BodyParameters
from pygarment.data_config import Properties

SIZES = {'s': 85, 'm': 94, 'l': 100}          # 마네킹 가슴둘레
WAIST = {'s': 69, 'm': 78, 'l': 84}
HIPS  = {'s': 91, 'm': 100, 'l': 106}
PANTS_WB_FACTOR = 81.5 / 78                   # 바지 허리밴드 내장 여유
BOTTOM = {                                    # (허리여유, 엉덩이여유)
    'pants_slacks': (2, None),
    'skirt_pencil': (2, 6),
    'dress_basic':  (None, 6),
}
EASE  = {'tshirt_basic': 14, 'shirt_slim': 8, 'shirt_over': 30,
         'dress_basic': 14, 'pants_slacks': None, 'skirt_pencil': None}

# --- 키별 등급화 -------------------------------------------------------------
# 길이는 마네킹의 _leg_length 에 비례합니다(pants.py:205, skirt_paneled.py:330).
# standard_{s,m,l} 은 셋 다 162cm 라 _leg_length 가 75.30 으로 같고, 그래서
# 지금까지 S/M/L 바지 길이가 동일했습니다. 아래 두 항목만 갈아끼워 키를 반영합니다.
#
# waist_line 은 둘레 버킷(B)이 아니라 키(H)에만 연동합니다. B 축을 따라가면
# H0B3(30.2)처럼 튀는 값이 길이로 새어 들어옵니다. 기준은 각 키의 B2 값입니다.
HEIGHT_CM     = {'H0': 154, 'H1': 162, 'H2': 170}
REF_WAISTLINE = {'H0': 35.6, 'H1': 37.1, 'H2': 38.6}   # 각 키의 B2 버킷 값
GRADED        = {'pants_slacks', 'skirt_pencil', 'dress_basic'}   # 하의 계열만
PANTS_LENGTH  = 0.75          # 다리길이 대비 비율. 0.3 이면 무릎 위에서 끝납니다.

# 같은 키라도 사이즈가 커지면 길이도 조금 깁니다. ARKET 여성 사이즈표의
# 다리안쪽길이가 S 78-79 / M 80-81 / L 82-83 으로 사이즈당 2cm 오릅니다.
# 다리길이 75cm 기준 0.02 가 약 1.5cm 이므로 사이즈당 그만큼 줍니다.
# 키 등급화와 달리 조합 수에는 영향이 없습니다.
SIZE_LEN_ADJ  = {'s': -0.02, 'm': 0.0, 'l': +0.02}

out_root = Path(Properties('./system.json')['output'])
index, eases = {}, {}

for tag, bust in SIZES.items():
    for dname, target in EASE.items():
        # 등급화 대상이 아니면 기존과 동일하게 162cm 한 벌만 만듭니다.
        htags = sorted(HEIGHT_CM) if dname in GRADED else ['*']
        for htag in htags:
            body = BodyParameters(f'./assets/bodies/standard_{tag}.yaml')
            if htag != '*':
                # 세로 항목만 주입합니다. 둘레를 격자 몸에서 가져오면
                # S/M/L 사이즈표가 격자 몸 치수로 덮여 깨집니다.
                body.params['height'] = HEIGHT_CM[htag]
                body.params['waist_line'] = REF_WAISTLINE[htag]
                body.eval_dependencies()

            d = yaml.safe_load(open(f'./assets/design_params/{dname}.yaml'))['design']
            if target is not None:
                d['shirt']['width']['v'] = (bust + target) / bust
            if dname in BOTTOM:
                we, he = BOTTOM[dname]
                W, H = WAIST[tag], HIPS[tag]
                if we is not None and 'waistband' in d:
                    base = W * PANTS_WB_FACTOR if dname.startswith('pants') else W
                    d['waistband']['waist']['v'] = (W + we) / base
                if he is not None and 'pencil-skirt' in d:
                    d['pencil-skirt']['flare']['v'] = (H + he) / H
            if dname == 'pants_slacks':
                d['pants']['length']['v'] = PANTS_LENGTH + SIZE_LEN_ADJ[tag]

            # 산출물 이름은 {design}_{size}__{bucket} 그대로입니다. 키는 버킷에
            # 이미 들어 있으므로(H0B2 의 H0) 패턴 이름에 붙이지 않습니다.
            # 붙이면 make_spec.py 의 rsplit('_', 1) 이 사이즈로 잘못 읽습니다.
            name = f'{dname}_{tag}'
            piece = MetaGarment(f'{name}_{htag}' if htag != '*' else name, body, d)
            pattern = piece.assembly()
            if piece.is_self_intersecting():
                print(f'  ! {name} [{htag}] self-intersecting')
            folder = pattern.serialize(out_root, tag='', to_subfolder=True,
                                       with_3d=False, with_text=False,
                                       view_ids=False, with_printable=False)
            spec = next(Path(folder).glob('*_specification.json'))
            index.setdefault(name, {})[htag] = str(spec)
            if target is not None:
                eases[name] = target
            print(f'  ok {name} [{htag}]'
                  + (f'  width={d["shirt"]["width"]["v"]:.3f}' if target else '')
                  + (f'  leg={body.params["_leg_length"]:.1f}' if dname in GRADED else ''))

Path('patterns_index.json').write_text(json.dumps(index, indent=2))
Path('ease_target.json').write_text(json.dumps(eases, indent=2))
print(f'\n{len(index)}종 / 패턴 {sum(len(v) for v in index.values())}벌 생성')
