"""Small statistics helpers (no scipy dependency)."""
from math import sqrt
from statistics import NormalDist

_N = NormalDist()


def z_crit(conf: float) -> float:
    return _N.inv_cdf(0.5 + conf / 2)


def two_prop(yt, nt, yc, nc, conf=0.95):
    """Difference in proportions (treated - control) with Wald CI and two-sided p-value."""
    pt, pc = yt / nt, yc / nc
    d = pt - pc
    # Agresti-Caffo adjustment (+1 success, +1 failure per arm) keeps the SE honest when a count is 0 or tiny
    at, ac = (yt + 1) / (nt + 2), (yc + 1) / (nc + 2)
    se = sqrt(at * (1 - at) / (nt + 2) + ac * (1 - ac) / (nc + 2))
    z = z_crit(conf)
    p = 2 * (1 - _N.cdf(abs(d / se))) if se > 0 else 1.0
    return dict(pt=pt, pc=pc, diff=d, se=se, lo=d - z * se, hi=d + z * se, p=p,
                rel=d / pc if pc > 0 else float("nan"),
                rel_lo=(d - z * se) / pc if pc > 0 else float("nan"),
                rel_hi=(d + z * se) / pc if pc > 0 else float("nan"))


def cace(itt, se_itt, e, se_e, conf=0.95):
    """Effect on the exposed (Wald / IV estimator): ITT / exposure-rate, delta-method CI."""
    r = itt / e
    se = sqrt((se_itt / e) ** 2 + (itt * se_e / e**2) ** 2)
    z = z_crit(conf)
    return dict(effect=r, se=se, lo=r - z * se, hi=r + z * se)


def n_per_arm(p0, mde_abs, alpha=0.05, power=0.8, ratio=1.0):
    """Sample size of the *smaller* arm needed to detect an absolute lift (two-sided z-test)."""
    p1 = p0 + mde_abs
    za, zb = _N.inv_cdf(1 - alpha / 2), _N.inv_cdf(power)
    var = p0 * (1 - p0) + p1 * (1 - p1) / ratio
    return (za + zb) ** 2 * var / mde_abs**2


def fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"
