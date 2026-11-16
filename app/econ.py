"""Economics of targeting policies, computed from hold-out experiment bins (no model output is trusted
without randomized outcomes: lift is always measured treated-vs-control on users the model never saw)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    outcome: str
    value: float        # $ per incremental outcome (conversion or visit)
    cost: float         # $ per exposed user (media cost actually incurred)
    audience: int       # users in the planned campaign audience
    conf: float


def rebin(bins: pd.DataFrame, step: int = 5, half: int | None = None) -> pd.DataFrame:
    """Aggregate the 1% score bins into `step`% bins (optionally one random half only)."""
    b = bins if half is None else bins[bins.half == half]
    b = b.assign(pct=(b.pct - 1) // step * step + step)
    return b.groupby("pct", as_index=False)[["n", "n_t", "n_c", "y_t", "y_c", "exp_t", "sum_u", "sum_pc"]].sum()


def cumulative(bins: pd.DataFrame) -> pd.DataFrame:
    """Cumulative (top-k% by score) lift per targeted user, exposure rate and standard error."""
    b = bins.sort_values("pct")
    nt, nc = b.n_t.cumsum().to_numpy(float), b.n_c.cumsum().to_numpy(float)
    yt, yc, et = b.y_t.cumsum().to_numpy(float), b.y_c.cumsum().to_numpy(float), b.exp_t.cumsum().to_numpy(float)
    pt, pc = yt / nt, yc / nc
    n_all = (b.n_t + b.n_c).sum()
    frac = (nt + nc) / n_all
    d = pd.DataFrame(dict(
        pct=b.pct.to_numpy(), frac=frac, lift=pt - pc, exp_rate=et / nt,
        se=np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc), pt=pt, pc=pc))
    origin = d.iloc[[0]].copy()
    origin[["pct", "frac"]] = 0                      # zero users targeted -> zero spend, zero incremental
    return pd.concat([origin, d], ignore_index=True)


def policy(cum: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Add $ columns scaled to the planned audience."""
    c = cum.copy()
    users = c.frac * a.audience
    c["users"] = users
    c["incremental"] = c.lift * users
    c["spend"] = c.exp_rate * a.cost * users
    c["revenue"] = c.incremental * a.value
    c["profit"] = c.revenue - c.spend
    c["iroas"] = np.where(c.spend > 0, c.revenue / c.spend, np.nan)
    c["icpa"] = np.where(c.incremental > 0, c.spend / c.incremental, np.nan)
    return c


def optimum(c: pd.DataFrame) -> int:
    """Index of the profit-maximising cut-off (0 spend counts as an option)."""
    return int(c.profit.idxmax()) if c.profit.max() > 0 else -1


def honest_gain(bins_long: pd.DataFrame, a: Assumptions, step: int = 5) -> dict:
    """Split-sample validation: choose the profit-maximising cut-off on one random half of the hold-out,
    then measure profit of that cut-off (vs targeting everyone) on the OTHER half. Averaged over both directions."""
    out = []
    for h in (0, 1):
        sel = policy(cumulative(rebin(bins_long, step, h)), a)
        ev = policy(cumulative(rebin(bins_long, step, 1 - h)), a)
        i = optimum(sel)
        prof = ev.profit.iloc[i] if i >= 0 else 0.0
        out.append(dict(frac=sel.frac.iloc[i] if i >= 0 else 0.0, profit=prof, all_profit=ev.profit.iloc[-1],
                        inc=ev.incremental.iloc[i] if i >= 0 else 0.0, spend=ev.spend.iloc[i] if i >= 0 else 0.0))
    m = {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    m["gain"] = m["profit"] - m["all_profit"]
    return m


def rulebook_validated(lv_long: pd.DataFrame, a: Assumptions) -> dict:
    """Choose which cohorts to suppress (negative profit per targeted user) on one half, evaluate on the other."""
    cols = ["n", "n_t", "n_c", "y_t", "y_c", "exp_t"]
    out = []
    for h in (0, 1):
        sel = lv_long[lv_long.half == h].groupby("leaf")[cols].sum()
        ev = lv_long[lv_long.half == 1 - h].groupby("leaf")[cols].sum()
        profit_user = (sel.y_t / sel.n_t - sel.y_c / sel.n_c) * a.value - (sel.exp_t / sel.n_t) * a.cost
        keep = ev[(profit_user >= 0).reindex(ev.index).fillna(False)]
        scale = a.audience / ev.n.sum()
        lift = keep.y_t / keep.n_t - keep.y_c / keep.n_c
        out.append(dict(users=keep.n.sum() * scale, inc=(lift * keep.n).sum() * scale,
                        spend=((keep.exp_t / keep.n_t) * a.cost * keep.n).sum() * scale))
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def strategy_row(name: str, users, incremental, spend, value) -> dict:
    revenue = incremental * value
    return dict(Strategy=name, Targeted=users, Spend=spend, Incremental=incremental, Revenue=revenue,
                Profit=revenue - spend, iROAS=revenue / spend if spend > 0 else np.nan,
                iCPA=spend / incremental if incremental > 0 else np.nan)


def decile_table(bins: pd.DataFrame, conf_z: float) -> pd.DataFrame:
    b = bins.copy()
    b["decile"] = (b.pct - 1) // 10 + 1
    g = b.groupby("decile").agg(n_t=("n_t", "sum"), n_c=("n_c", "sum"), y_t=("y_t", "sum"), y_c=("y_c", "sum"),
                                exp_t=("exp_t", "sum"), n=("n", "sum"), sum_u=("sum_u", "sum")).reset_index()
    g["pred"] = g.sum_u / g.n
    pt, pc = g.y_t / g.n_t, g.y_c / g.n_c
    g["lift"] = pt - pc
    g["se"] = np.sqrt(pt * (1 - pt) / g.n_t + pc * (1 - pc) / g.n_c)
    g["lo"], g["hi"] = g.lift - conf_z * g.se, g.lift + conf_z * g.se
    g["exp_rate"] = g.exp_t / g.n_t
    return g
