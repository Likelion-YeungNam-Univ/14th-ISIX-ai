import json
from pathlib import Path

TOL_BY_FIT = {
    '슬림':  {'bust': (-2, 3),  'waist': (-2, 3),  'hips': (-2, 3),  'shoulder': (-1, 1)},
    '레귤러': {'bust': (-4, 6),  'waist': (-4, 6),  'hips': (-4, 6),  'shoulder': (-1, 2)},
    '오버핏': {'bust': (-6, 12), 'waist': (-6, 12), 'hips': (-6, 12), 'shoulder': (-1, 4)},
}
LABEL = {'bust':'가슴', 'waist':'허리', 'hips':'엉덩이', 'shoulder':'어깨'}
SPEC = json.loads(Path('garment_spec.json').read_text())

def judge_one(user, key):
    """user: {'bust':92,'waist':74,...}  key: 'tshirt_basic_m'"""
    g = SPEC[key]
    parts, worst = {}, 0.0
    for part, gcm in g['garment_cm'].items():
        if part not in user: continue
        actual = round(gcm - user[part], 1)          # 실제 여유
        dev = round(actual - g['ref_ease_cm'][part], 1)  # 편차
        lo, hi = TOL_BY_FIT[g['fit']][part]
        if dev < lo:   v, color = '꽉 낌', 'red'
        elif dev > hi: v, color = '여유 있음', 'blue'
        else:          v, color = '적정', 'green'
        parts[part] = {'label': LABEL[part], 'garment_cm': gcm,
                       'actual_ease': actual, 'ref_ease': g['ref_ease_cm'][part],
                       'deviation': dev, 'verdict': v, 'color': color}
        over = max(lo - dev, dev - hi, 0)
        worst = max(worst, over)
    total = round(sum(abs(p['deviation']) for p in parts.values()), 1)
    return {'garment': key, 'size': g['size'], 'fit': g['fit'],
            'parts': parts, 'penalty': round(worst, 1), 'total_dev': total,
            'wearable': all(p['verdict'] != '꽉 낌' for p in parts.values())}

def recommend(user, design):
    cands = [judge_one(user, f'{design}_{s}') for s in ('s','m','l')]
    ok = [c for c in cands if c['wearable']] or cands
    best = min(ok, key=lambda c: (c['penalty'], c['total_dev']))
    return best, cands

if __name__ == '__main__':
    user = {'bust': 88, 'waist': 72, 'hips': 95, 'shoulder': 38}
    print(f'사용자: {user}\n')
    for design in ['tshirt_basic','shirt_slim','shirt_over',
                   'dress_basic','pants_slacks','skirt_pencil']:
        best, cands = recommend(user, design)
        print(f"[{design}] → {best['size']} 추천  ({best['fit']})")
        for c in cands:
            marks = '  '.join(
                f"{p['label']} {p['deviation']:+5.1f} {p['verdict']}"
                for p in c['parts'].values())
            star = '★' if c is best else ' '
            print(f"  {star} {c['size']}: {marks}")
        print()
