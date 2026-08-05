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

out_root = Path(Properties('./system.json')['output'])
index, eases = {}, {}

for tag, bust in SIZES.items():
    body = BodyParameters(f'./assets/bodies/standard_{tag}.yaml')
    for dname, target in EASE.items():
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
        name = f'{dname}_{tag}'
        piece = MetaGarment(name, body, d)
        pattern = piece.assembly()
        if piece.is_self_intersecting():
            print(f'  ! {name} self-intersecting')
        folder = pattern.serialize(out_root, tag='', to_subfolder=True,
                                   with_3d=False, with_text=False,
                                   view_ids=False, with_printable=False)
        spec = next(Path(folder).glob('*_specification.json'))
        index[name] = str(spec)
        if target is not None:
            eases[name] = target
        print(f'  ok {name}' + (f'  width={d["shirt"]["width"]["v"]:.3f}' if target else ''))

Path('patterns_index.json').write_text(json.dumps(index, indent=2))
Path('ease_target.json').write_text(json.dumps(eases, indent=2))
print(f'\n{len(index)}개 패턴 생성')
