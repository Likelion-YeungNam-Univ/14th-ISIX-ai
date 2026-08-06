import json, re, subprocess, time, shutil
from pathlib import Path

BODIES = ['mean_all', 'mean_female']   # 팀원 체형 오면 여기 12개로 교체
TIMEOUT = 300

patterns = json.loads(Path('patterns_index.json').read_text())
out_dir = Path('Sim_results'); out_dir.mkdir(exist_ok=True)
res_file = Path('batch_results.json')
results = json.loads(res_file.read_text()) if res_file.exists() else {}

jobs = [(p, b) for b in BODIES for p in patterns]
print(f'총 {len(jobs)}개 (완료 {len(results)}개)\n')

for i, (pname, body) in enumerate(jobs, 1):
    key = f'{pname}__{body}'
    if key in results:
        continue
    t0 = time.time()
    try:
        r = subprocess.run(
            ['python', 'batch_sim.py', '-p', patterns[pname], '-b', body],
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
    if objs and time.time() - objs[-1].stat().st_mtime < TIMEOUT:
        shutil.copy(objs[-1], out_dir / f'{key}.obj')
        rec['obj'] = f'Sim_results/{key}.obj'

    results[key] = rec
    res_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    flag = 'X' if rec['fail'] else 'o'
    print(f"[{i}/{len(jobs)}] {flag} {key}  관통={rec['body_int']} {rec['sec']}s")

fails = [k for k, v in results.items() if v['fail']]
print(f"\n완료 {len(results)}개 / 실패 {len(fails)}개")
for k in fails: print('  실패:', k)
