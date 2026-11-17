"""Criteo Incrementality Desk — a Streamlit + DuckDB decision tool for a performance-marketing agency.

Client: the agency's CFO-facing account team.  Question: "Which of our ad spend is truly incremental,
and who should we stop targeting?"  Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import econ as E  # noqa: E402
import stats as S  # noqa: E402
import ui  # noqa: E402
from ui import (BAD, CONTROL, GOOD, GRID, INK, MUTED, SEG_BLURB, SEG_COLORS, TREAT, WARN,  # noqa: E402
                money, num, pct)

st.set_page_config(page_title="Incrementality Desk", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

# ------------------------------------------------------------------ sidebar: global assumptions
with st.sidebar:
    st.markdown("### ⚙️ Assumptions")
    st.caption("The dataset has no prices. Edit these to match the client's economics; every page recalculates.")
    outcome = st.radio("Outcome to optimise", ["conversion", "visit"], horizontal=True,
                       help="Conversion = purchase (rare, 0.3%). Visit = site visit (4.7%), a denser proxy.")
    default_value = 40.0 if outcome == "conversion" else 2.0
    value = st.number_input(f"Value per incremental {outcome} ($)", 0.01, 10_000.0, default_value, 1.0,
                            key=f"value_{outcome}")
    cost = st.number_input("Media cost per exposed user ($)", 0.01, 100.0, 1.00, 0.10,
                           help="Cost is only incurred when the ad is actually delivered (exposure=1).")
    audience = st.number_input("Planned campaign audience (users)", 100_000, 1_000_000_000, 10_000_000, 1_000_000)
    conf = st.select_slider("Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
                            format_func=lambda x: f"{int(x * 100)}%")
    st.divider()
    st.caption("Data: Criteo Uplift v2.1 · 13.98M users · randomized A/B (85% treatment / 15% control).\n\n"
               "Engine: DuckDB · Models: gradient-boosted T-learner trained on 70% of users, evaluated on a 30% hold-out.")

A = E.Assumptions(outcome, value, cost, int(audience), conf)
Z = S.z_crit(conf)
OUT = outcome

# ------------------------------------------------------------------ core numbers
arms = D.arms()
yt, yc = arms[f"{OUT}_t"], arms[f"{OUT}_c"]
itt = S.two_prop(yt, arms["nt"], yc, arms["nc"], conf)
exp_rate = arms["n_exposed"] / arms["nt"]
exp_se = (exp_rate * (1 - exp_rate) / arms["nt"]) ** 0.5
cace = S.cace(itt["diff"], itt["se"], exp_rate, exp_se, conf)
incr_total = itt["diff"] * arms["nt"]                 # incremental outcomes caused by the campaign
claimed = arms[f"{OUT}_exposed"]                      # what an exposure-based report would credit
share_true = incr_total / claimed if claimed else float("nan")

bins_long = D.uplift_bins(OUT)
bins = E.rebin(bins_long, 1)                       # 1% bins (all hold-out users) for deciles
bins5 = E.rebin(bins_long, 5)                      # 5% bins for the policy curve (less noisy)
cum = E.cumulative(bins5)
hg = E.honest_gain(bins_long, A)
lv_long = D.leaves(OUT)
rb = E.rulebook_validated(lv_long, A)               # cost-aware cohort rulebook, validated out-of-sample
pol = E.policy(cum, A)
opt_i = E.optimum(pol)
all_row = pol.iloc[-1]

ui.hero(
    "Incrementality Desk",
    "Which ad spend actually changes customer behaviour, and which users should we stop paying to reach? "
    "A decision tool for the performance-agency account team, built on 14M randomized Criteo users.",
    ["Client: Performance agency → skeptical CFO", "Mandate: prove incremental lift & target on uplift",
     "Opportunity: Do-Not-Target engine", "DuckDB · 13.98M rows"])

tabs = st.tabs(["📋 Executive Brief", "🧪 1 · Incrementality Ledger", "🎯 2 · Who To Target",
                "🚫 3 · Do-Not-Target Engine", "💰 4 · Business Case", "🤖 AI Analyst", "🔎 Methods & SQL"])

# =================================================================== TAB 0: EXECUTIVE BRIEF
with tabs[0]:
    ui.section("The one-minute answer", f"Outcome: <b>{OUT}</b> · confidence {int(conf * 100)}% · all values update with the sidebar")
    c = st.columns(5)
    with c[0]:
        ui.kpi("Users in experiment", num(arms["nt"] + arms["nc"]), f"{num(arms['nt'])} treated · {num(arms['nc'])} control")
    with c[1]:
        ui.kpi("Measured lift", f"+{itt['diff'] * 100:.3f} pp",
               f"{pct(itt['pc'], 3)} → {pct(itt['pt'], 3)} · {itt['rel'] * 100:+.0f}% relative", "blue")
    with c[2]:
        ui.kpi("Statistical certainty", S.fmt_p(itt["p"]), f"p-value · CI {itt['lo'] * 100:.3f} to {itt['hi'] * 100:.3f} pp",
               "good" if itt["p"] < 0.05 else "bad")
    with c[3]:
        ui.kpi("Truly incremental", f"{share_true * 100:.0f}%", f"of the {OUT}s an exposure report would claim", "bad" if share_true < 0.75 else "good")
    with c[4]:
        ui.kpi("Profit gain from uplift targeting", money(hg["gain"]),
               f"out-of-sample, vs. target-all · {num(A.audience)} users", "good" if hg["gain"] > 0 else "bad")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        anyway = max(claimed - incr_total, 0)
        fig = go.Figure()
        fig.add_bar(name="Truly incremental", x=["Exposure-based report<br>(what gets claimed)", "Randomized test<br>(what is true)"],
                    y=[incr_total, incr_total], marker_color=GOOD, hovertemplate="%{y:,.0f} incremental<extra></extra>")
        fig.add_bar(name="Would have happened anyway", x=["Exposure-based report<br>(what gets claimed)", "Randomized test<br>(what is true)"],
                    y=[anyway, 0], marker_color=CONTROL, hovertemplate="%{y:,.0f} would have happened anyway<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.45, title=f"{OUT.title()}s credited to the campaign")
        fig.add_annotation(x=0, y=claimed, text=f"<b>{num(claimed)}</b>", showarrow=False, yshift=14)
        fig.add_annotation(x=1, y=incr_total, text=f"<b>{num(incr_total)}</b>", showarrow=False, yshift=14)
        ui.show(ui.style(fig, 400, True, f"{OUT}s"))
    with right:
        f = go.Figure()
        f.add_scatter(x=pol.frac * 100, y=pol.profit, mode="lines", line=dict(color=TREAT, width=3), name="Target by predicted uplift",
                      hovertemplate="Target top %{x:.0f}%<br>Profit %{y:$,.0f}<extra></extra>")
        f.add_hline(y=all_row.profit, line=dict(color=CONTROL, dash="dash"),
                    annotation_text=f"Target everyone: {money(all_row.profit)}", annotation_position="bottom right")
        if opt_i >= 0:
            f.add_scatter(x=[pol.frac.iloc[opt_i] * 100], y=[pol.profit.iloc[opt_i]], mode="markers", name="Optimum",
                          marker=dict(color=GOOD, size=13, line=dict(color="#fff", width=2)),
                          hovertemplate="Optimum: top %{x:.0f}%<br>%{y:$,.0f}<extra></extra>")
        ui.show(ui.style(f, 400, True, "Profit ($)", "% of audience targeted (highest uplift first)").update_layout(
            title=f"Profit on a {num(A.audience)}-user campaign"))

    st.write("")
    ui.section("What we recommend")
    if opt_i >= 0:
        o = pol.iloc[opt_i]
        rec = (f"Target the <b>top {o.frac * 100:.0f}%</b> of users by predicted uplift: {money(o.profit)} profit, "
               f"iROAS <b>{o.iroas:.1f}×</b> versus {all_row.iroas:.1f}× when targeting everyone "
               f"({money(all_row.profit)} profit), while capturing <b>{o.incremental / all_row.incremental * 100:.0f}%</b> "
               f"of all incremental {OUT}s with {o.spend / all_row.spend * 100:.0f}% of the media spend.")
    else:
        rec = "At these economics no segment is profitable; lower the cost or revisit the offer."
    ui.callout("1 · Move budget to persuadable users", rec, "good")
    ui.callout("2 · Report incrementality, not attribution",
               f"Exposed users produced {num(claimed)} {OUT}s, but a randomized holdout shows only <b>{num(incr_total)}</b> "
               f"({share_true * 100:.0f}%) were caused by the ads. The rest would have happened anyway. "
               f"Effect on exposed users: {cace['effect'] * 100:.2f} pp (CI {cace['lo'] * 100:.2f} to {cace['hi'] * 100:.2f}).", "warn")
    ui.callout("3 · Small lift, large dollars",
               f"A {itt['diff'] * 100:.3f} pp lift looks trivial, but on {num(A.audience)} users it is "
               f"<b>{num(itt['diff'] * A.audience)}</b> extra {OUT}s = {money(itt['diff'] * A.audience * A.value)} of revenue "
               f"(CI {money(itt['lo'] * A.audience * A.value)} to {money(itt['hi'] * A.audience * A.value)}).")

# =================================================================== TAB 1: INCREMENTALITY LEDGER
with tabs[1]:
    ui.section("1 · Incrementality Ledger", "Randomization lets us separate what the ads caused from what would have happened anyway.")
    c = st.columns(4)
    with c[0]:
        ui.kpi("Control rate", pct(itt["pc"], 3), f"{num(arms['nc'])} users held out", "")
    with c[1]:
        ui.kpi("Treated rate", pct(itt["pt"], 3), f"{num(arms['nt'])} users eligible for ads", "blue")
    with c[2]:
        ui.kpi("Intent-to-treat lift", f"{itt['diff'] * 100:+.3f} pp",
               f"{itt['rel'] * 100:+.1f}% relative [{itt['rel_lo'] * 100:.1f}%, {itt['rel_hi'] * 100:.1f}%]", "good")
    with c[3]:
        ui.kpi("Lift among exposed (CACE)", f"{cace['effect'] * 100:+.2f} pp",
               f"{pct(exp_rate, 1)} of treated were exposed · CI [{cace['lo'] * 100:.2f}, {cace['hi'] * 100:.2f}]", "good")
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=["Control (no ads)", "Treated (ads eligible)"], y=[itt["pc"] * 100, itt["pt"] * 100],
                    marker_color=[CONTROL, TREAT], width=0.5,
                    error_y=dict(type="data", array=[Z * (itt["pc"] * (1 - itt["pc"]) / arms["nc"]) ** 0.5 * 100,
                                                     Z * (itt["pt"] * (1 - itt["pt"]) / arms["nt"]) ** 0.5 * 100],
                                 color=INK, thickness=1.5, width=6),
                    hovertemplate="%{x}<br>%{y:.4f}%<extra></extra>", text=[pct(itt["pc"], 3), pct(itt["pt"], 3)],
                    textposition="outside")
        ui.show(ui.style(fig, 380, False, f"{OUT.title()} rate (%)").update_layout(
            title=f"Randomized A/B result · {int(conf * 100)}% CI"))
    with c2:
        # attribution illusion
        p_e = arms[f"{OUT}_exposed"] / arms["n_exposed"]
        p_u = arms[f"{OUT}_unexposed"] / arms["n_unexposed"]
        cf = p_e - cace["effect"]
        labels = ["Control<br>(everyone)", "Treated but<br>not exposed", "Exposed<br>(observed)", "Exposed<br>(counterfactual*)"]
        vals = [itt["pc"], p_u, p_e, cf]
        colors = [CONTROL, CONTROL, TREAT, "#BFD3FB"]
        fig = go.Figure(go.Bar(x=labels, y=[v * 100 for v in vals], marker_color=colors, width=0.55,
                               text=[pct(v, 2) for v in vals], textposition="outside",
                               hovertemplate="%{x}<br>%{y:.3f}%<extra></extra>"))
        fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.93, xanchor="left", showarrow=False, align="left",
                           font=dict(color=BAD, size=14),
                           text=f"Naive lift: <b>{p_e / p_u:.0f}×</b><br>True lift: <b>{p_e / cf:.1f}×</b>")
        ui.show(ui.style(fig, 380, False, f"{OUT.title()} rate (%)").update_layout(title="The attribution illusion"))
    ui.callout("Why exposure-based reports overstate impact",
               f"Exposed users convert at {pct(p_e, 2)} versus {pct(p_u, 3)} for unexposed treated users, a naive <b>{p_e / p_u:.0f}×</b> lift. "
               f"But users who get exposed are the active, high-intent ones. The randomized control group shows that without the ad they would still have "
               f"reached ≈{pct(cf, 2)} (*counterfactual = exposed rate − CACE). The true lift is <b>{p_e / cf:.1f}×</b>, "
               f"so about <b>{(1 - share_true) * 100:.0f}%</b> of credited {OUT}s are not incremental.", "warn")

    c1, c2 = st.columns([1, 1])
    with c1:
        ui.section("Does the evidence get more certain with scale?", "Same experiment, nested random subsamples.")
        cv = D.convergence()
        rows = []
        for _, r in cv.iterrows():
            k = "visit" if OUT == "visit" else "conv"
            x = S.two_prop(r[f"{k}_t"], r.nt, r[f"{k}_c"], r.nc, conf)
            rows.append(dict(n=r.sample_n, lift=x["diff"] * 100, lo=x["lo"] * 100, hi=x["hi"] * 100))
        cvd = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_scatter(x=cvd.n, y=cvd.hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=cvd.n, y=cvd.lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(37,99,235,.15)",
                        name=f"{int(conf * 100)}% CI", hoverinfo="skip")
        fig.add_scatter(x=cvd.n, y=cvd.lift, mode="lines+markers", line=dict(color=TREAT, width=3), marker=dict(size=9),
                        name="Estimated lift (pp)", hovertemplate="n = %{x:,.0f}<br>lift %{y:.3f} pp<extra></extra>")
        fig.add_hline(y=0, line=dict(color=INK, width=1))
        fig.update_xaxes(type="log", tickvals=[1e4, 1e5, 1e6, 1e7], ticktext=["10K", "100K", "1M", "10M"])
        ui.show(ui.style(fig, 340, True, "Lift (percentage points)", "Users in experiment (log scale)"))
        small = cvd.iloc[1]
        st.caption(f"With {num(small.n)} users the interval is {small.lo:.2f} to {small.hi:.2f} pp"
                   f"{' (cannot rule out zero)' if small.lo < 0 < small.hi else ''}. "
                   f"At full scale it is {cvd.iloc[-1].lo:.3f} to {cvd.iloc[-1].hi:.3f} pp.")
    with c2:
        ui.section("Randomization health check", "If assignment were biased, treated and control would differ on user features.")
        bal = D.balance()
        fig = go.Figure(go.Bar(x=bal.feature, y=bal.smd, marker_color=[GOOD if abs(v) < 0.1 else BAD for v in bal.smd],
                               hovertemplate="%{x}: SMD %{y:.4f}<extra></extra>"))
        fig.add_hrect(y0=-0.1, y1=0.1, fillcolor="rgba(5,150,105,.08)", line_width=0)
        fig.update_yaxes(range=[-0.15, 0.15])
        ui.show(ui.style(fig, 340, False, "Standardized mean difference"))
        st.caption(f"Largest |SMD| = {bal.smd.abs().max():.4f} (rule of thumb: < 0.10 is balanced). The 85/15 split is a design choice, not a bias.")

    with st.expander("🧮 Design the next test: how many users do we need?"):
        c = st.columns(4)
        base = c[0].number_input("Baseline rate (%)", 0.001, 50.0, round(itt["pc"] * 100, 3), 0.01, format="%.3f")
        mde = c[1].number_input("Smallest lift worth detecting (pp)", 0.001, 10.0, 0.05 if OUT == "conversion" else 0.3, 0.01, format="%.3f")
        pw = c[2].select_slider("Power", [0.7, 0.8, 0.9], 0.8, format_func=lambda x: f"{int(x * 100)}%")
        share = c[3].slider("Share assigned to control (%)", 5, 50, 15)
        n_ctrl = S.n_per_arm(base / 100, mde / 100, 1 - conf, pw, ratio=(100 - share) / share)
        total = n_ctrl / (share / 100)
        st.markdown(f"**≈ {num(n_ctrl)} control users** and **{num(total)} users in total** are required "
                    f"to detect a {mde:.3f} pp lift on a {base:.3f}% baseline. "
                    f"The full dataset ({num(arms['nt'] + arms['nc'])} users) is {(arms['nt'] + arms['nc']) / total:.1f}× that.")

# =================================================================== TAB 2: WHO TO TARGET
with tabs[2]:
    ui.section("2 · Who to target", "Uplift model (T-learner, gradient boosting) scores each user by predicted <b>incremental</b> response. "
                                    "Everything below is measured on 4.2M hold-out users the model never saw, using the randomized control group.")
    dec = E.decile_table(bins, Z)
    cap30 = cum.loc[cum.pct == 30, "lift"].iloc[0] * 0.30 / cum.lift.iloc[-1]
    cap50 = cum.loc[cum.pct == 50, "lift"].iloc[0] * 0.50 / cum.lift.iloc[-1]
    # Qini-style area vs random
    I = cum.lift * cum.frac
    qini = float(np.trapezoid(I - cum.frac * cum.lift.iloc[-1], cum.frac))
    c = st.columns(4)
    with c[0]:
        ui.kpi("Top 20% of users capture", pct(cum.loc[cum.pct == 20, "lift"].iloc[0] * 0.2 / cum.lift.iloc[-1], 0),
               f"of all incremental {OUT}s (random = 20%)", "good")
    with c[1]:
        ui.kpi("Top 50% capture", pct(cap50, 0), "of all incremental outcomes (random = 50%)", "good")
    with c[2]:
        top_d, bot_d = dec.iloc[0], dec.iloc[-1]
        ui.kpi("Top vs bottom decile lift", f"{top_d.lift * 100:.2f} vs {bot_d.lift * 100:.2f} pp",
               "measured, not predicted", "blue")
    with c[3]:
        ui.kpi("Top 10% capture", pct(cum.loc[cum.pct == 10, "lift"].iloc[0] * 0.1 / cum.lift.iloc[-1], 0),
               f"of all incremental {OUT}s (random = 10%)", "good")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        hi = (I + Z * cum.se * cum.frac) * 1000
        lo = (I - Z * cum.se * cum.frac) * 1000
        fig.add_scatter(x=cum.frac * 100, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=cum.frac * 100, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(37,99,235,.15)",
                        name=f"{int(conf * 100)}% CI", hoverinfo="skip")
        fig.add_scatter(x=[0, 100], y=[0, cum.lift.iloc[-1] * 1000], mode="lines", name="Random targeting",
                        line=dict(color=CONTROL, dash="dash", width=2), hoverinfo="skip")
        fig.add_scatter(x=cum.frac * 100, y=I * 1000, mode="lines", name="Uplift model", line=dict(color=TREAT, width=3),
                        hovertemplate="Target top %{x:.0f}%<br>%{y:.2f} incremental per 1,000 users<extra></extra>")
        ui.show(ui.style(fig, 380, True, f"Incremental {OUT}s per 1,000 audience users", "% of audience targeted").update_layout(
            title="Cumulative uplift (Qini) curve · hold-out"))
    with c2:
        fig = go.Figure()
        fig.add_bar(x=[f"D{int(d)}" for d in dec.decile], y=dec.lift * 100, name="Measured lift (pp)",
                    marker_color=[GOOD if v > 0 else BAD for v in dec.lift], width=0.6,
                    error_y=dict(type="data", array=(Z * dec.se * 100).tolist(), color=INK, thickness=1.2, width=4),
                    hovertemplate="Decile %{x}<br>measured %{y:.3f} pp<extra></extra>")
        fig.add_scatter(x=[f"D{int(d)}" for d in dec.decile], y=dec.pred * 100, mode="markers", name="Model-predicted",
                        marker=dict(symbol="diamond", size=10, color=INK, line=dict(color="#fff", width=1.5)),
                        hovertemplate="Decile %{x}<br>predicted %{y:.3f} pp<extra></extra>")
        ui.show(ui.style(fig, 380, True, "Lift (pp)", "Predicted-uplift decile (D1 = most persuadable)").update_layout(
            title="Do the model's rankings hold up? Measured lift by decile"))

    ui.section("Policy simulator", "Choose how much of the audience to target. Profit uses your sidebar economics.")
    default_f = int(round(pol.frac.iloc[opt_i] * 100)) if opt_i >= 0 else 100
    f_sel = st.slider("Share of audience to target (highest predicted uplift first)", 5, 100, max(5, int(round(default_f / 5) * 5)),
                      step=5, help="Default = profit-maximising cut-off.")
    row = pol.iloc[int(np.argmin(np.abs(pol.pct - f_sel)))]
    c = st.columns(5)
    with c[0]:
        ui.kpi("Users targeted", num(row.users), f"{f_sel}% of {num(A.audience)}")
    with c[1]:
        ui.kpi("Incremental " + OUT + "s", num(row.incremental), f"{row.incremental / all_row.incremental * 100:.0f}% of the maximum", "blue")
    with c[2]:
        ui.kpi("Media spend", money(row.spend), f"{row.spend / all_row.spend * 100:.0f}% of blast-all spend")
    with c[3]:
        ui.kpi("Profit", money(row.profit), f"vs {money(all_row.profit)} target-all", "good" if row.profit > all_row.profit else "bad")
    with c[4]:
        ui.kpi("iROAS / iCPA", f"{row.iroas:.2f}× / {money(row.icpa, 2)}", f"all-in: {all_row.iroas:.2f}× / {money(all_row.icpa, 2)}")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=pol.frac * 100, y=pol.profit, mode="lines", line=dict(color=TREAT, width=3), name="Profit",
                        hovertemplate="Target %{x:.0f}% → %{y:$,.0f}<extra></extra>")
        fig.add_hline(y=all_row.profit, line=dict(color=CONTROL, dash="dash"), annotation_text="Target everyone")
        fig.add_vline(x=f_sel, line=dict(color=INK, width=1, dash="dot"))
        if opt_i >= 0:
            fig.add_scatter(x=[pol.frac.iloc[opt_i] * 100], y=[pol.profit.iloc[opt_i]], mode="markers", name="Optimum",
                            marker=dict(color=GOOD, size=13, line=dict(color="#fff", width=2)))
        ui.show(ui.style(fig, 360, True, "Profit ($)", "% of audience targeted").update_layout(title="Profit curve"))
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=pol.frac * 100, y=pol.revenue, mode="lines", name="Incremental revenue", line=dict(color=GOOD, width=3),
                        hovertemplate="%{x:.0f}% → %{y:$,.0f}<extra>Revenue</extra>")
        fig.add_scatter(x=pol.frac * 100, y=pol.spend, mode="lines", name="Media spend", line=dict(color=WARN, width=3),
                        hovertemplate="%{x:.0f}% → %{y:$,.0f}<extra>Spend</extra>")
        fig.add_vline(x=f_sel, line=dict(color=INK, width=1, dash="dot"))
        ui.show(ui.style(fig, 360, True, "$", "% of audience targeted").update_layout(title="Where the curves diverge, stop buying"))
    ui.callout("Out-of-sample check (no peeking)",
               f"We pick the best cut-off on one random half of the hold-out and score it on the other half. "
               f"The cut-off (≈ top <b>{hg['frac'] * 100:.0f}%</b>) earns <b>{money(hg['profit'])}</b> on unseen users versus "
               f"<b>{money(hg['all_profit'])}</b> for targeting everyone, a validated gain of <b>{money(hg['gain'])}</b>. "
               f"The final proof is always a live A/B of the policy itself.", "good" if hg["gain"] > 0 else "warn")

# =================================================================== TAB 3: DO NOT TARGET
with tabs[3]:
    ui.section("3 · Do-Not-Target engine", "The product feature for retargeting platforms: a ranked list of who to suppress, with plain-English rules and measured proof.")
    top_pct = st.slider("Treat the top X% by predicted uplift as 'Persuadables'", 5, 40, 20, 5)
    seg = D.segments(OUT, top_pct)
    seg["rate_t"], seg["rate_c"] = seg.y_t / seg.n_t, seg.y_c / seg.n_c
    seg["lift"] = seg.rate_t - seg.rate_c
    seg["se"] = np.sqrt(seg.rate_t * (1 - seg.rate_t) / seg.n_t + seg.rate_c * (1 - seg.rate_c) / seg.n_c)
    seg["exp_rate"] = seg.exp_t / seg.n_t
    seg["profit_user"] = seg.lift * A.value - seg.exp_rate * A.cost
    seg["share"] = seg.n / seg.n.sum()
    order = ["Persuadables", "Sure things", "Lost causes", "Predicted backfire"]
    seg = seg.set_index("segment").reindex([s for s in order if s in set(seg.segment)]).reset_index()

    c1, c2 = st.columns([1.15, 1])
    with c1:
        sm = D.scatter_sample(OUT, top_pct)
        fig = go.Figure()
        for sname in order:
            d = sm[sm.segment == sname]
            fig.add_scattergl(x=d.base_rate * 100, y=d.uplift * 100, mode="markers", name=sname,
                              marker=dict(color=SEG_COLORS[sname], size=5, opacity=0.55),
                              hovertemplate=f"{sname}<br>baseline %{{x:.3f}}%<br>uplift %{{y:.3f}} pp<extra></extra>")
        fig.add_hline(y=0, line=dict(color=INK, width=1))
        fig.update_layout(legend=dict(itemsizing="constant"))
        ui.show(ui.style(fig, 420, True, "Predicted uplift (pp)", f"Baseline {OUT} rate without ads (%)").update_layout(
            title="Four personas · 12,000-user sample"))
    with c2:
        fig = go.Figure(go.Bar(
            y=seg.segment[::-1], x=(seg.lift * 100)[::-1], orientation="h", width=0.6,
            marker_color=[SEG_COLORS[s] for s in seg.segment[::-1]],
            error_x=dict(type="data", array=(Z * seg.se * 100)[::-1].tolist(), color=INK, thickness=1.2, width=4),
            hovertemplate="%{y}<br>measured lift %{x:.3f} pp<extra></extra>"))
        ui.show(ui.style(fig, 420, False, None, "Measured lift on hold-out (pp)").update_layout(
            title="Proof: realized lift by persona", yaxis_title=None))

    # persona cards
    cols = st.columns(len(seg))
    for col, (_, r) in zip(cols, seg.iterrows()):
        with col:
            lo, hi = r.lift - Z * r.se, r.lift + Z * r.se
            tone = "good" if lo > 0 else ("bad" if hi < 0 else "")
            ui.kpi(r.segment, f"{r.lift * 100:+.3f} pp", f"{r.share * 100:.0f}% of users · CI [{lo * 100:.2f}, {hi * 100:.2f}]", tone)
            blurb = SEG_BLURB[r.segment]
            if r.segment == "Predicted backfire":
                blurb = ("✅ Confirmed: ads reduce response here. Never target." if hi < 0 else
                         "⚠️ Hold-out does NOT confirm backfire (lift is not negative). Treated as low-priority, not harmful.")
            st.caption(blurb)

    supp = seg[seg.segment.isin(["Lost causes", "Predicted backfire"]) | (seg.profit_user < 0)].segment.tolist()
    ui.section("Suppression list: who not to pay for", "Rulebook: each rule is a plain-English cohort with its measured economics. "
                                                         "🔴 = unprofitable at your assumptions, 🟡 = unproven, 🟢 = keep.")
    rules = D.rules_meta()
    key = "visit" if OUT == "visit" else "conv"
    lv_long = D.leaves(OUT)
    lv = lv_long.groupby("leaf", as_index=False)[["n", "n_t", "n_c", "y_t", "y_c", "exp_t", "sum_u"]].sum()
    lv["pred_uplift"] = lv.sum_u / lv.n
    lv = lv.sort_values("pred_uplift", ascending=False).reset_index(drop=True)
    lv["Rule"] = lv.leaf.astype(str).map(lambda k: " AND ".join(rules[key][k]) if rules[key][k] else "all users")
    lv["rate_t"], lv["rate_c"] = lv.y_t / lv.n_t, lv.y_c / lv.n_c
    lv["lift"] = lv.rate_t - lv.rate_c
    lv["se"] = np.sqrt(lv.rate_t * (1 - lv.rate_t) / lv.n_t + lv.rate_c * (1 - lv.rate_c) / lv.n_c)
    lv["exp_rate"] = lv.exp_t / lv.n_t
    lv["profit_user"] = lv.lift * A.value - lv.exp_rate * A.cost
    lv["lo"], lv["hi"] = lv.lift - Z * lv.se, lv.lift + Z * lv.se

    def decide(r):
        if r.profit_user < 0:
            return "🔴 Suppress"
        return "🟢 Target" if r.lo * A.value - r.exp_rate * A.cost > 0 else "🟡 Test / hold"
    lv["Decision"] = lv.apply(decide, axis=1)
    scale = A.audience / lv.n.sum()
    lv["users_campaign"] = lv.n * scale
    lv["profit_total"] = lv.profit_user * lv.users_campaign
    show_df = pd.DataFrame({
        "Decision": lv.Decision, "Rule (who)": lv.Rule, "Share of users (%)": lv.n / lv.n.sum() * 100,
        "Measured lift (pp)": lv.lift * 100,
        f"{int(conf * 100)}% CI (pp)": [f"{l * 100:.3f} to {h * 100:.3f}" for l, h in zip(lv.lo, lv.hi)],
        "Exposure rate (%)": lv.exp_rate * 100, "Profit / targeted user ($)": lv.profit_user,
        "Profit in campaign ($)": lv.profit_total})
    st.dataframe(show_df, hide_index=True, width="stretch", height=min(60 + 36 * len(show_df), 640), column_config={
        "Share of users (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100.0),
        "Measured lift (pp)": st.column_config.NumberColumn(format="%.3f"),
        "Exposure rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Profit / targeted user ($)": st.column_config.NumberColumn(format="$%.4f"),
        "Profit in campaign ($)": st.column_config.NumberColumn(format="$%,.0f")})
    saved_spend = all_row.spend - rb["spend"]
    lost_inc = all_row.incremental - rb["inc"]
    net = (rb["inc"] * A.value - rb["spend"]) - all_row.profit
    n_supp = A.audience - rb["users"]
    c = st.columns(4)
    with c[0]:
        ui.kpi("Users suppressed", num(n_supp), f"{n_supp / A.audience * 100:.0f}% of the audience", "bad")
    with c[1]:
        ui.kpi("Media spend avoided", money(saved_spend), "no ads served to suppressed users", "good")
    with c[2]:
        ui.kpi(f"Incremental {OUT}s forgone", num(max(lost_inc, 0)), f"{max(lost_inc, 0) / all_row.incremental * 100:.0f}% of total" if all_row.incremental else "")
    with c[3]:
        ui.kpi("Net profit gain", money(net), "validated out-of-sample", "good" if net > 0 else "bad")
    st.download_button("⬇️ Download Do-Not-Target rulebook (CSV)", show_df.to_csv(index=False).encode(), "do_not_target_rulebook.csv", "text/csv")

    c1, c2 = st.columns([1, 1])
    with c1:
        pf = D.profile_top_bottom(OUT, 20)
        fig = go.Figure()
        fig.add_bar(y=pf.feature, x=pf.top, orientation="h", name="Top 20% (persuadable)", marker_color=GOOD,
                    hovertemplate="%{y}: %{x:.2f} SD<extra>Top 20%</extra>")
        fig.add_bar(y=pf.feature, x=pf.bottom, orientation="h", name="Bottom 20% (do not target)", marker_color=BAD,
                    hovertemplate="%{y}: %{x:.2f} SD<extra>Bottom 20%</extra>")
        fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
        ui.show(ui.style(fig, 430, True, None, "Feature mean vs. audience average (standard deviations)").update_layout(
            title="What separates who to target from who to skip"))
    with c2:
        imp = pd.Series(rules[f"importance_{key}"]).sort_values()
        fig = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h", marker_color=TREAT, width=0.65,
                               hovertemplate="%{y}: %{x:.5f}<extra></extra>"))
        ui.show(ui.style(fig, 430, False, None, "Permutation importance (log-loss increase)").update_layout(title="Which signals drive response"))
    st.caption("Features f0–f11 are anonymized by Criteo, so rules are expressed on those fields. The client platform maps them to real attributes.")

# =================================================================== TAB 4: BUSINESS CASE
with tabs[4]:
    ui.section("4 · Business case", f"Three strategies on a {num(A.audience)}-user campaign. All outcomes are measured on hold-out experiment data, scaled to your audience.")
    rows = [E.strategy_row("A · Target everyone", all_row.users, all_row.incremental, all_row.spend, A.value),
            E.strategy_row(f"B · Uplift top {hg['frac'] * 100:.0f}%", hg["frac"] * A.audience, hg["inc"], hg["spend"], A.value),
            E.strategy_row("C · Rulebook (cost-aware)", rb["users"], rb["inc"], rb["spend"], A.value)]
    sdf = pd.DataFrame(rows)
    best = sdf.Profit.idxmax()
    cards = st.columns(len(sdf))
    for i, (col, (_, r)) in enumerate(zip(cards, sdf.iterrows())):
        with col:
            ui.kpi(r.Strategy, money(r.Profit), f"iROAS {r.iROAS:.2f}× · iCPA {money(r.iCPA, 2)} · spend {money(r.Spend)}",
                   "good" if i == best else "")
    st.caption("B and C are validated out-of-sample: the cut-off (B) or the suppressed cohorts (C) are chosen on one random half of the hold-out "
               "and scored on the other half. C wins because it also looks at *exposure cost*: some high-uplift cohorts are expensive to reach.")
    st.write("")
    fig = go.Figure()
    fig.add_bar(x=sdf.Strategy, y=sdf.Revenue, name="Incremental revenue", marker_color=GOOD, width=0.3,
                hovertemplate="%{x}<br>Revenue %{y:$,.0f}<extra></extra>")
    fig.add_bar(x=sdf.Strategy, y=sdf.Spend, name="Media spend", marker_color=WARN, width=0.3,
                hovertemplate="%{x}<br>Spend %{y:$,.0f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.4)
    for _, r in sdf.iterrows():
        fig.add_annotation(x=r.Strategy, y=max(r.Revenue, r.Spend), text=f"<b>Profit {money(r.Profit)}</b>", showarrow=False, yshift=16)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        ui.show(ui.style(fig, 380, True, "$").update_layout(title="Revenue vs spend by strategy"))
    with c2:
        out = sdf.copy()
        out["Targeted"] = out.Targeted.map(num)
        for ccol in ["Spend", "Revenue", "Profit"]:
            out[ccol] = out[ccol].map(money)
        out["Incremental"] = out.Incremental.map(num)
        out["iROAS"] = out.iROAS.map(lambda v: f"{v:.2f}×")
        out["iCPA"] = out.iCPA.map(lambda v: money(v, 2))
        out["Strategy"] = ["A · Everyone", "B · Uplift cut-off", "C · Rulebook"]
        st.dataframe(out.set_index("Strategy").T, width="stretch", height=320)

    ui.section("How sensitive is the answer to your assumptions?", "Extra profit from uplift targeting (B) vs blasting everyone (A), re-optimised at each point.")
    cmul = [0.25, 0.5, 1, 1.5, 2, 3, 4]
    vmul = [0.5, 0.75, 1, 1.5, 2]
    z = np.zeros((len(vmul), len(cmul)))
    for i, vm in enumerate(vmul):
        for j, cm in enumerate(cmul):
            a2 = E.Assumptions(OUT, A.value * vm, A.cost * cm, A.audience, conf)
            p2 = E.policy(cum, a2)
            z[i, j] = max(p2.profit.max(), 0) - p2.profit.iloc[-1]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"${A.cost * m:.2f}" for m in cmul], y=[f"${A.value * m:,.2f}" for m in vmul],
        text=[[money(v) for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, "#EFF6FF"], [1, TREAT]], xgap=3, ygap=3,
        hovertemplate="Cost/exposure %{x}<br>Value %{y}<br>Gain %{z:$,.0f}<extra></extra>", colorbar=dict(title="$ gain")))
    ui.style(fig, 380, False, f"Value per {OUT}", "Media cost per exposed user").update_layout(
        title="Profit gain of uplift targeting vs target-all, across assumptions")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    ui.show(fig)

    ui.section("Why a 0.3% effect is worth millions", "Scale calculator: incremental lift × audience × value, with statistical uncertainty.")
    sizes = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    cal = pd.DataFrame({
        "Audience": [num(s) for s in sizes],
        f"Incremental {OUT}s": [num(itt["diff"] * s) for s in sizes],
        "Revenue (point estimate)": [money(itt["diff"] * s * A.value) for s in sizes],
        f"Revenue range ({int(conf * 100)}% CI)": [f"{money(itt['lo'] * s * A.value)} – {money(itt['hi'] * s * A.value)}" for s in sizes],
        "Media spend": [money(exp_rate * A.cost * s) for s in sizes],
        "Net profit": [money((itt["diff"] * A.value - exp_rate * A.cost) * s) for s in sizes]})
    st.dataframe(cal, hide_index=True, width="stretch")
    st.caption("Spend assumes the observed exposure rate of the treated group. Revenue uses the intent-to-treat lift for the full audience.")

# =================================================================== TAB 5: AI ANALYST
with tabs[5]:
    import analyst as AI

    ui.section("Ask the Analyst",
               "Ask in plain English. Gemini can only answer by calling the same functions that draw this dashboard, "
               "and every figure it quotes is checked against those results.")
    tb = AI.Toolbox(A)
    with st.expander("How the AI works", expanded=False):
        st.markdown(
            "1. Your question goes to **Google Gemini** with 5 *tools*: incrementality, targeting policy, strategy "
            "comparison, segments, model validation.\n"
            "2. Gemini **cannot compute numbers itself**. It calls the tools; our code runs them on the DuckDB tables "
            "using your sidebar assumptions.\n"
            "3. Gemini writes the answer from the returned numbers.\n"
            "4. A **figure check** pulls every number out of the answer and verifies it against the tool outputs. "
            "Anything that doesn't match is flagged ⚠️.")

    ui.callout("Auto-brief at your current assumptions", "Generated from the same tools, no AI key needed.", "good")
    st.markdown(AI.deterministic_brief(tb))

    key = AI.api_key()
    if not key:
        st.text_input("Google Gemini API key (kept in this browser session only)", type="password", key="gemini_key",
                      help="Free key at aistudio.google.com/apikey. Or set GEMINI_API_KEY in Streamlit secrets.")
        key = AI.api_key()
    if not key:
        st.info("Add a free Gemini API key above to ask questions. The auto-brief above always works.", icon="🔑")
    else:
        st.caption(f"Model: `{AI.model_name()}`")
        st.session_state.setdefault("chat", [])

        def render(ans):
            st.markdown(ans["text"])
            g = ans["grounding"]
            label = ("No figures quoted" if not g.total else
                     f"✅ All {g.total} figures traced to the data" if g.ok == g.total else
                     f"⚠️ {g.total - g.ok} of {g.total} figures could not be traced. Treat them with caution.")
            with st.expander(label):
                if g.checked:
                    st.write(" · ".join(f"{'✅' if ok else '⚠️'} `{t}`" for t, ok in g.checked))
                for name, inp, out in ans["calls"]:
                    st.markdown(f"**{name}** `{inp}`")
                    st.json(out, expanded=False)

        suggestions = ["Should the CFO renew the campaign as-is?",
                       "What if the value per conversion is $60?",
                       "Which users should we stop paying to reach?",
                       "Which strategy makes the most profit, and why?",
                       "How sure are we the model's ranking is real?"]
        picked = None
        for col, s_ in zip(st.columns(len(suggestions)), suggestions):
            if col.button(s_, use_container_width=True, key=f"sugg_{s_}"):
                picked = s_
        for turn in st.session_state.chat:
            with st.chat_message(turn["role"]):
                render(turn["answer"]) if turn["role"] == "assistant" else st.markdown(turn["content"])
        with st.form("ask_form", clear_on_submit=True):
            typed = st.text_input("Your question", placeholder="e.g. What's our profit if cost per exposed user drops to $0.50?")
            sent = st.form_submit_button("Ask", type="primary")
        q = (typed if sent and typed.strip() else None) or picked
        if q:
            with st.chat_message("user"):
                st.markdown(q)
            history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat][-6:]
            with st.chat_message("assistant"):
                with st.status("Querying the experiment…", expanded=True) as status:
                    try:
                        ans = AI.ask(q, history, tb, key, on_step=status.write)
                    except Exception as e:
                        status.update(label="The AI call failed", state="error", expanded=True)
                        msg = str(e)
                        st.error("Gemini's free-tier rate limit was hit. Wait a minute and ask again."
                                 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg)
                                 else f"The AI call failed ({type(e).__name__}: {msg[:300]}).")
                        ans = None
                    if ans:
                        status.update(label=f"Answered with {ans['model']} · {len(ans['calls'])} data lookups",
                                      state="complete", expanded=False)
                if ans:
                    render(ans)
                    st.session_state.chat += [{"role": "user", "content": q},
                                              {"role": "assistant", "content": ans["text"], "answer": ans}]

# =================================================================== TAB 6: METHODS & SQL
with tabs[6]:
    ui.section("Methods, data and SQL", "Everything here is reproducible: DuckDB holds the 14M-row table and every figure comes from a query.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
**Data** · Criteo Uplift Prediction v2.1 (Hugging Face `criteo/criteo-uplift`, CC BY-NC-SA 4.0). {num(arms['nt'] + arms['nc'])} users from randomized ad
experiments: `treatment` (eligible for ads), `exposure` (ad actually delivered), `visit`, `conversion`, 12 anonymized features.

**Pipeline** (`pipeline/`)
1. `01_ingest.py`: CSV.gz → DuckDB table `uplift` (columnar, 14M rows, ~75 s) + hash-based 70/30 train/test split.
2. `02_train_score.py`: T-learner (two gradient-boosted classifiers per outcome: treated and control). Trained on the 70% train split only; scores the 30% hold-out. A depth-4 surrogate tree turns the model into readable rules.
3. `03_aggregates.py`: randomization balance table and nested subsamples.
4. `04_build_app_db.py`: a few-MB database of pre-aggregated results, so the dashboard can be deployed without the 750 MB file.

**Statistics**
- Intent-to-treat lift = treated rate − control rate (two-proportion CI with the Agresti–Caffo small-count adjustment).
- Effect on the exposed (CACE) = ITT ÷ exposure rate. Valid because control users cannot be exposed (one-sided non-compliance) and non-exposed treated users are unaffected (exclusion restriction).
- Targeting results use only hold-out users; lift of any user group is always treated-vs-control *within that group*.
""")
    with c2:
        st.markdown("""
**Limitations to state out loud**
- Prices are not in the data; value per outcome and cost per exposure are editable assumptions.
- Features are anonymized, so rules cannot be mapped to real attributes without the client.
- Chosen cut-offs are tuned on the hold-out; a final A/B of the policy itself is the real test.
- One experiment window; no time dimension, so seasonality and ad fatigue are unobserved.
- Per-cohort lift on rare conversions is noisy; intervals are shown wherever a decision depends on a sign.
- The optional Criteo Attribution dataset (campaign-level budget reallocation) is not used in this version.
""")
        fs = D.feature_summary()
        st.dataframe(fs, hide_index=True, width="stretch", height=300, column_config={
            "min": st.column_config.NumberColumn(format="%.2f"), "mean": st.column_config.NumberColumn(format="%.2f"),
            "max": st.column_config.NumberColumn(format="%.2f")})

    if D.FULL:
        ui.section("Live SQL console", "Read-only access to the DuckDB database. Tables: uplift (14M rows), scored (4.2M hold-out), balance, convergence.")
        examples = {
            "Experiment summary by arm": "SELECT treatment, count(*) AS users, avg(exposure) AS exposure_rate, avg(visit) AS visit_rate, avg(conversion) AS conv_rate\nFROM uplift GROUP BY 1 ORDER BY 1",
            "Lift by predicted-uplift decile (hold-out)": "SELECT decile, avg(u_conv)*100 AS predicted_pp,\n  (avg(conversion) FILTER (WHERE treatment=1) - avg(conversion) FILTER (WHERE treatment=0))*100 AS measured_pp\nFROM (SELECT *, ntile(10) OVER (ORDER BY u_conv DESC) AS decile FROM scored)\nGROUP BY 1 ORDER BY 1",
            "Exposed vs unexposed (selection bias)": "SELECT treatment, exposure, count(*) AS users, avg(conversion)*100 AS conv_pct\nFROM uplift GROUP BY 1,2 ORDER BY 1,2",
            "Table schema": "DESCRIBE scored",
        }
    else:
        ui.section("Live SQL console", "Read-only DuckDB. The deployed build holds pre-aggregated results of the 14M-row experiment (arm_cells, bins, seg_stats, leaves, balance, convergence) "
                                       "plus a 100,000-user hold-out sample (scored_sample).")
        examples = {
            "Experiment summary by arm (all 14M users)": "SELECT treatment, sum(n) AS users, sum(visits)*1.0/sum(n) AS visit_rate, sum(conv)*1.0/sum(n) AS conv_rate\nFROM arm_cells GROUP BY 1 ORDER BY 1",
            "Exposed vs unexposed (selection bias)": "SELECT treatment, exposure, n AS users, conv*100.0/n AS conv_pct\nFROM arm_cells ORDER BY 1,2",
            "Lift by predicted-uplift decile (100K sample)": "SELECT decile, avg(u_conv)*100 AS predicted_pp,\n  (avg(conversion) FILTER (WHERE treatment=1) - avg(conversion) FILTER (WHERE treatment=0))*100 AS measured_pp\nFROM (SELECT *, ntile(10) OVER (ORDER BY u_conv DESC) AS decile FROM scored_sample)\nGROUP BY 1 ORDER BY 1",
            "Table schema": "DESCRIBE scored_sample",
        }
    pick = st.selectbox("Example queries", list(examples))
    sql = st.text_area("SQL", examples[pick], height=140, key=f"sql_{pick}")
    if st.button("▶ Run query", type="primary"):
        try:
            import time
            t0 = time.perf_counter()
            res = D.run_sql(sql)
            st.success(f"{len(res):,} rows in {(time.perf_counter() - t0) * 1000:.0f} ms")
            st.dataframe(res, width="stretch", hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
