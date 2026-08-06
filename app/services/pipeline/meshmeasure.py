"""메시 단면 계측 원시함수. 아바타(AI-1)와 의류 시뮬(AI-2)이 함께 쓴다.

여유량 = 옷 둘레 - 몸 둘레 이므로, 두 값이 **같은 방법으로** 계산되지 않으면
여유량 자체가 무의미해진다. 각자 구현하면 볼록껍질을 쓰냐 마냐, 겹친 원단 중
어느 루프를 잡냐에서 갈리고, 그 차이는 에러 없이 여유량에 그대로 실린다.
그래서 계측의 수학은 이 파일 하나로 모으고 양쪽이 import 해서 쓴다.

좌표계 전제 (body_meta.json / body_grid.json 과 동일):
    단위 m / Y 위 / +Z 정면 / 발바닥 y=0

둘레는 볼록껍질(convex hull) 둘레로 잰다.
  몸: 줄자는 팽팽히 당겨져 오목한 곳(가슴 사이, 척추 홈)으로 파고들지 않는다.
  옷: 스펙표의 실측은 옷을 평평히 펴서 재므로 주름을 타고 넘지 않는다.
     시뮬 결과 메시의 주름을 원시 둘레로 재면 실제보다 크게 나온다.
  -> 양쪽 모두 볼록껍질이 맞다. 원시 둘레도 함께 반환하니 차이를 확인할 것.
"""
import numpy as np
from scipy.spatial import ConvexHull

# 단면에서 이보다 작은 조각은 버린다 [m].
# SMPL-X 는 watertight 가 아니라 눈알/입 안쪽 공동이 있고, 평면이 그런 걸
# 스치면 둘레 6cm 짜리 퇴화 루프가 나온다. 옷 메시도 시접/라벨 조각이 나온다.
MIN_PERIM = 0.12
MIN_PERIM_LIMB = 0.08          # 팔/다리는 더 가늘다


def plane_basis(normal):
    """평면 위의 직교 기저 두 개."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def section_loops(mesh, origin, normal, min_pts=4):
    """평면으로 자른 단면의 폐곡선들. 사지나 겹친 원단이 있으면 여러 개가 나온다."""
    try:
        sec = mesh.section(plane_origin=np.asarray(origin, float),
                           plane_normal=np.asarray(normal, float))
    except Exception:
        return []
    if sec is None:
        return []
    return [np.asarray(p, float) for p in sec.discrete if len(p) >= min_pts]


def perimeters(loop, normal):
    """(원시 둘레, 볼록껍질 둘레) [m]. 판정에는 볼록껍질 쪽을 쓴다."""
    u, v = plane_basis(normal)
    P = np.stack([loop @ u, loop @ v], 1)
    raw = float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum())
    if not np.allclose(loop[0], loop[-1]):
        raw += float(np.linalg.norm(loop[0] - loop[-1]))
    try:
        h = P[ConvexHull(P).vertices]
    except Exception:
        return raw, raw
    hull = float(np.linalg.norm(np.diff(np.vstack([h, h[:1]]), axis=0), axis=1).sum())
    return raw, hull


def contains(loop, point, normal):
    """2D 투영 후 점이 폐곡선 내부인지 (ray casting)."""
    u, w = plane_basis(normal)
    P = np.stack([loop @ u, loop @ w], 1)
    q = np.array([point @ u, point @ w])
    a, b = P, np.roll(P, -1, axis=0)
    straddle = (a[:, 1] > q[1]) != (b[:, 1] > q[1])
    if not straddle.any():
        return False
    a, b = a[straddle], b[straddle]
    x = a[:, 0] + (q[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool((x > q[0]).sum() % 2 == 1)


def pick_loop(loops, point, normal=(0, 1, 0), min_perim=MIN_PERIM, prefer="outer"):
    """측정 대상 폐곡선을 고른다.

    "축에서 가장 가까운 것" 만 보면 안 된다. 퇴화 조각이 최솟값 탐색에서
    목둘레로 뽑히는 사고가 실제로 났다.

    prefer:
      "outer" - 축을 감싸는 것 중 가장 큰 것. 몸, 그리고 옷의 바깥면.
      "inner" - 축을 감싸는 것 중 가장 작은 것.
                옷은 플래킷/칼라/밑단 접힘 때문에 같은 높이에 원단이 겹쳐
                동심 루프가 두 개 이상 나온다. 여유량은 몸에 닿는 안쪽 면
                기준이므로, 옷 단면은 "inner" 가 맞는 경우가 많다.
                어느 쪽을 쓸지는 옷 종류를 보고 AI-2 가 정할 것.
    """
    if not loops:
        return None
    point = np.asarray(point, float)
    normal = np.asarray(normal, float)
    big = [L for L in loops if perimeters(L, normal)[0] >= min_perim]
    if not big:
        return None
    inside = [L for L in big if contains(L, point, normal)]
    if inside:
        key = (lambda L: perimeters(L, normal)[1])
        return max(inside, key=key) if prefer == "outer" else min(inside, key=key)
    return min(big, key=lambda L: np.linalg.norm(L.mean(0) - point))


def girth_at(mesh, y, axis_xz=(0.0, 0.0), min_perim=MIN_PERIM, prefer="outer"):
    """높이 y 의 수평 단면 둘레 [m]. 없으면 None.

    여유량 계산의 공용 진입점이다. 옷과 몸을 **같은 y** 로 호출해서 빼면
    그것이 그 높이의 둘레 여유량이 된다.

        ease_cm = (girth_at(garment, y) - girth_at(body, y)) * 100

    KD-Tree 최근접 거리로 재지 말 것. 그건 반경 간격이라 둘레 여유량과
    약 2pi 배 차이가 난다. 스펙표는 둘레 단위로 적히므로 비교가 성립하지 않는다.
    (반경 간격은 히트맵 렌더링용으로만 쓰고, 색 경계는 둘레 기준을 2pi 로 나눠 환산)
    """
    n = (0.0, 1.0, 0.0)
    loops = section_loops(mesh, [0.0, y, 0.0], n)
    L = pick_loop(loops, np.array([axis_xz[0], y, axis_xz[1]]), n, min_perim, prefer)
    return None if L is None else perimeters(L, n)[1]


def girth_perp(mesh, A, B, t, min_perim=MIN_PERIM_LIMB, prefer="outer"):
    """A->B 축의 t 위치에서 축에 수직인 단면 둘레 [m]. 팔/다리·소매·바지통용.

    기울어진 사지를 수평으로 자르면 1/cos 만큼 부풀려진다.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = B - A
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    d = d / n
    P = A + t * (B - A)
    L = pick_loop(section_loops(mesh, P, d), P, d, min_perim, prefer)
    return None if L is None else perimeters(L, d)[1]


def radial_clearance(garment_v, body_kdtree):
    """정점별 반경 간격 [m]. 히트맵 렌더링 전용.

    판정과 사이즈 추천에는 쓰지 말 것 (girth_at 주석 참고).
    """
    d, _ = body_kdtree.query(np.asarray(garment_v, float))
    return d


def girth_to_radial(girth_cm):
    """둘레 여유량 -> 반경 간격 환산 [cm]. 원통 근사.

    히트맵 색 경계를 둘레 기준 판정과 1:1 로 맞출 때 쓴다.
    둘레 +8cm 는 반경으로 약 1.27cm 다.
    """
    return girth_cm / (2.0 * np.pi)
