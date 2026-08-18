import json, re, subprocess, sys, time, shutil
from pathlib import Path

BODIES = ['H0B0', 'H0B1', 'H0B2', 'H0B3', 'H1B0', 'H1B1', 'H1B2', 'H1B3', 'H2B0', 'H2B1', 'H2B2', 'H2B3']   # body_grid.json 12구간
# 길이를 늘린 바지는 정점이 7,084 -> 12,336 으로 늘어 시뮬이 느려집니다.
# 300 이면 실측 264~328초 구간과 겹쳐서 절반 가까이 타임아웃으로 잘립니다.
TIMEOUT = 900

patterns = json.loads(Path('patterns_index.json').read_text())
out_dir = Path('Sim_results'); out_dir.mkdir(exist_ok=True)
res_file = Path('batch_results.json')
results = json.loads(res_file.read_text()) if res_file.exists() else {}

# 키별 패턴은 자기 키의 버킷에만 씁니다. H0B2 는 H0 패턴, H2B3 은 H2 패턴.
# 등급화하지 않은 품목은 '*' 한 벌을 12구간 전부에 씁니다. 조합 수는 그대로입니다.
jobs = []
for b in BODIES:
    for p, paths in patterns.items():
        spec = paths.get(b[:2]) or paths.get('*')
        if spec is None:
            print(f'  ! {p} 에 {b[:2]} 패턴이 없어 건너뜁니다')
            continue
        jobs.append((p, b, spec))
print(f'총 {len(jobs)}개 (완료 {len(results)}개)\n')

for i, (pname, body, spec) in enumerate(jobs, 1):
    key = f'{pname}__{body}'
    if key in results:
        continue
    t0 = time.time()
    try:
        r = subprocess.run(
            # conda 를 활성화하지 않고 돌리면 'python' 이 PATH 에 없습니다.
            # 지금 이 스크립트를 실행 중인 인터프리터를 그대로 씁니다.
            [sys.executable, 'batch_sim.py', '-p', spec, '-b', body, '--smpl'],
            capture_output=True, text=True, timeout=TIMEOUT)
        log = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        log = 'TIMEOUT'

    def grab(pat, cast=int, default=None):
        m = re.search(pat, log)
        return cast(m.group(1)) if m else default

    rec = {
        'body_int': grab(r'BODY CLOTH INTERSECTIONS:\s+(\d+)'),
        'self_int': grab(r'Self-Intersecting with (\d+)'),
        'fail': 'is fail: True' in log or log == 'TIMEOUT',
        'frames': grab(r'#frames=(\d+)'),
        'sec': round(time.time() - t0, 1),
    }

    objs = sorted(Path('Logs').rglob('*_sim.obj'), key=lambda p: p.stat().st_mtime)
    if objs and objs[-1].stat().st_mtime > t0:
        shutil.copy(objs[-1], out_dir / f'{key}.obj')
        rec['obj'] = f'Sim_results/{key}.obj'

    results[key] = rec
    res_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    flag = 'X' if rec['fail'] else 'o'
    print(f"[{i}/{len(jobs)}] {flag} {key}  관통={rec['body_int']} {rec['sec']}s")

fails = [k for k, v in results.items() if v['fail']]
print(f"\n완료 {len(results)}개 / 실패 {len(fails)}개")
for k in fails: print('  실패:', k)
