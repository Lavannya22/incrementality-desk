"""AI analyst: plain-English Q&A over the Incrementality Desk, grounded in the app's own data.

How it works
1. The question goes to Google Gemini together with a list of *tools* (function declarations).
2. Gemini never sees raw rows and never does the maths. To answer, it must call the same functions that
   draw the dashboard (data.py / econ.py / stats.py); our code runs them on the DuckDB tables and returns numbers.
3. Gemini writes the answer from those numbers.
4. `grounding_check` extracts every figure from the answer and verifies it against the tool outputs,
   so made-up numbers are flagged in the UI.

Without an API key, `deterministic_brief` narrates the same tool outputs from a template.
Free key: https://aistudio.google.com/apikey (free tier, no billing needed).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, replace

import data as D
import econ as E
import stats as S

DEFAULT_MODEL = "gemini-3.8-flash"


# --------------------------------------------------------------------------- tools
_ECON = {
    "value": {"type": "number", "description": "$ per incremental outcome; omit to use the sidebar value"},
    "cost": {"type": "number", "description": "$ media cost per exposed user; omit to use the sidebar value"},
    "audience": {"type": "integer", "description": "planned campaign audience (users); omit to use the sidebar value"},
}

TOOLS = [
    {"name": "get_incrementality",
     "description": "The randomized A/B result for the whole experiment: control vs treated rate, lift with CI and p-value, "
                    "effect on exposed users (CACE), how many outcomes an exposure-based report would claim vs how many "
                    "were truly incremental, and revenue of the lift on the planned audience. Use for 'do the ads work', "
                    "'is it significant', 'how much is attribution overstating'.",
     "parameters": {"type": "object", "properties": dict(_ECON)}},
    {"name": "get_targeting_policy",
     "description": "Profit, spend, incremental outcomes, iROAS and iCPA when targeting only the top X% of users by "
                    "predicted uplift (X = 10..100 in steps of 10), the profit-maximising cut-off, and the out-of-sample "
                    "validated gain vs targeting everyone. Use for 'who should we target', 'what if we only target the top N%'.",
     "parameters": {"type": "object", "properties": dict(_ECON)}},
    {"name": "compare_strategies",
     "description": "Business case: A = target everyone, B = uplift top-X% cut-off, C = cost-aware cohort rulebook. "
                    "Users targeted, spend, incremental outcomes, revenue, profit, iROAS, iCPA. B and C are validated out-of-sample.",
     "parameters": {"type": "object", "properties": dict(_ECON)}},
    {"name": "get_segments",
     "description": "Four personas (Persuadables, Sure things, Lost causes, Predicted backfire) with share of users and "
                    "measured lift with CI on hold-out users. top_pct = share of users treated as Persuadables (default 20).",
     "parameters": {"type": "object", "properties": {"top_pct": {"type": "integer", "minimum": 5, "maximum": 50}}}},
    {"name": "get_model_validation",
     "description": "Does the uplift model's ranking hold on users it never saw? Measured vs predicted lift per "
                    "predicted-uplift decile (D1 = most persuadable) and the share of incremental outcomes captured by the top 10/20/50%.",
     "parameters": None},
]


def _r(x, sig: int = 6):
    """Round floats (for the LLM and for grounding) without losing meaning; drop NaN/inf."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return x
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            x = float(x)
    except Exception:
        pass
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(f"{x:.{sig}g}")
    if isinstance(x, dict):
        return {k: _r(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, sig) for v in x]
    return x


class Toolbox:
    """Runs the dashboard's own calculations for the LLM, with the sidebar assumptions as defaults."""

    def __init__(self, a: E.Assumptions):
        self.a = a

    def _a(self, args: dict) -> E.Assumptions:
        a = self.a
        if args.get("value"):
            a = replace(a, value=float(args["value"]))
        if args.get("cost"):
            a = replace(a, cost=float(args["cost"]))
        if args.get("audience"):
            a = replace(a, audience=int(args["audience"]))
        return a

    def run(self, name: str, args: dict) -> dict:
        a = self._a(args or {})
        out = a.outcome
        assumptions = {"outcome": out, "value_per_outcome": a.value, "cost_per_exposed_user": a.cost,
                       "audience": a.audience, "confidence": a.conf}

        if name == "get_incrementality":
            arms = D.arms()
            itt = S.two_prop(arms[f"{out}_t"], arms["nt"], arms[f"{out}_c"], arms["nc"], a.conf)
            er = arms["n_exposed"] / arms["nt"]
            cace = S.cace(itt["diff"], itt["se"], er, (er * (1 - er) / arms["nt"]) ** 0.5, a.conf)
            incr = itt["diff"] * arms["nt"]
            claimed = arms[f"{out}_exposed"]
            return _r({"assumptions": assumptions,
                       "users_treated": arms["nt"], "users_control": arms["nc"],
                       "control_rate": itt["pc"], "treated_rate": itt["pt"],
                       "lift": itt["diff"], "lift_ci": [itt["lo"], itt["hi"]], "relative_lift": itt["rel"],
                       "p_value": itt["p"], "exposure_rate": er,
                       "effect_on_exposed_cace": cace["effect"], "cace_ci": [cace["lo"], cace["hi"]],
                       "claimed_by_exposure_report": claimed, "truly_incremental": incr,
                       "share_truly_incremental": incr / claimed if claimed else None,
                       "incremental_on_audience": itt["diff"] * a.audience,
                       "revenue_on_audience": itt["diff"] * a.audience * a.value,
                       "revenue_on_audience_ci": [itt["lo"] * a.audience * a.value, itt["hi"] * a.audience * a.value]})

        if name == "get_targeting_policy":
            bins_long = D.uplift_bins(out)
            pol = E.policy(E.cumulative(E.rebin(bins_long, 10)), a)
            hg = E.honest_gain(bins_long, a)
            pol5 = E.policy(E.cumulative(E.rebin(bins_long, 5)), a)
            i = E.optimum(pol5)
            rows = pol[pol.frac > 0][["frac", "users", "incremental", "spend", "revenue", "profit", "iroas", "icpa"]]
            rows = rows.assign(top_pct=(rows.frac * 100).round().astype(int)).drop(columns="frac")
            return _r({"assumptions": assumptions, "rows": rows.to_dict("records"),
                       "optimum_top_pct": round(float(pol5.frac.iloc[i]) * 100) if i >= 0 else 0,
                       "optimum_profit": float(pol5.profit.iloc[i]) if i >= 0 else 0.0,
                       "target_everyone_profit": float(pol5.profit.iloc[-1]),
                       "validated_gain_vs_everyone": hg["gain"], "validated_cutoff_top_pct": hg["frac"] * 100})

        if name == "compare_strategies":
            bins_long = D.uplift_bins(out)
            all_row = E.policy(E.cumulative(E.rebin(bins_long, 5)), a).iloc[-1]
            hg = E.honest_gain(bins_long, a)
            rb = E.rulebook_validated(D.leaves(out), a)
            rows = [E.strategy_row("A · Target everyone", all_row.users, all_row.incremental, all_row.spend, a.value),
                    E.strategy_row(f"B · Uplift top {hg['frac'] * 100:.0f}%", hg["frac"] * a.audience, hg["inc"], hg["spend"], a.value),
                    E.strategy_row("C · Rulebook (cost-aware)", rb["users"], rb["inc"], rb["spend"], a.value)]
            return _r({"assumptions": assumptions, "strategies": rows})

        if name == "get_segments":
            top = int(args.get("top_pct") or 20)
            seg = D.segments(out, top)
            res = []
            for r in seg.itertuples():
                t = S.two_prop(r.y_t, r.n_t, r.y_c, r.n_c, a.conf)
                res.append({"segment": r.segment, "share_of_users": r.n / seg.n.sum(), "measured_lift": t["diff"],
                            "lift_ci": [t["lo"], t["hi"]], "predicted_uplift": r.pred_uplift})
            return _r({"assumptions": assumptions, "top_pct": top, "segments": res})

        if name == "get_model_validation":
            bins = E.rebin(D.uplift_bins(out), 1)
            dec = E.decile_table(bins, S.z_crit(a.conf))
            cum = E.cumulative(E.rebin(D.uplift_bins(out), 10))
            total = cum.lift.iloc[-1] * cum.frac.iloc[-1]
            capture = {f"top_{int(round(f * 100))}pct": float(l * f / total)
                       for f, l in zip(cum.frac, cum.lift) if round(f * 100) in (10, 20, 50)}
            return _r({"deciles": dec[["decile", "lift", "lo", "hi", "pred"]].rename(
                        columns={"lift": "measured_lift", "lo": "ci_low", "hi": "ci_high", "pred": "predicted_lift"}
                       ).to_dict("records"), "share_of_incremental_captured": capture})

        raise ValueError(f"unknown tool {name}")


# --------------------------------------------------------------------------- grounding check
NUM_RE = re.compile(r"(?<![\w.])[-−+]?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|pp|×|x\b|[KMB]\b|k\b|m\b)?")


@dataclass
class Grounding:
    checked: list = field(default_factory=list)

    @property
    def ok(self) -> int:
        return sum(1 for _, g in self.checked if g)

    @property
    def total(self) -> int:
        return len(self.checked)


def _flatten(obj, out: list) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))


def grounding_check(text: str, tool_outputs: list, context_numbers: list) -> Grounding:
    """Every figure in the answer must match a tool output (or a stated assumption) within its displayed precision."""
    vals: list = []
    for o in tool_outputs:
        _flatten(o, vals)
    base = set(context_numbers)
    for v in vals:
        base.update({v, abs(v)})
        if abs(v) < 1:          # rates/shares: allow percent / percentage-point form (0.00115 -> 0.115 pp)
            base.add(abs(v) * 100)
    g = Grounding()
    for m in NUM_RE.finditer(text):
        whole, dec, unit = m.group(1), m.group(2) or "", (m.group(3) or "").strip()
        raw = m.group(0).strip()
        x = float(whole.replace(",", "") + dec)
        if not unit and not dec and x <= 12 and "$" not in raw:
            continue  # ordinals, deciles, "D1", "3 strategies"
        if 1990 <= x <= 2100 and not unit:
            continue  # years
        mult = {"K": 1e3, "k": 1e3, "M": 1e6, "m": 1e6, "B": 1e9}.get(unit, 1.0)
        value = x * mult
        step = (10 ** -len(dec[1:]) if dec else 1) * mult
        hit = any(abs(abs(value) - abs(t)) <= max(step * 0.51, abs(t) * 0.006) for t in base)
        g.checked.append((raw, hit))
    return g


# --------------------------------------------------------------------------- LLM loop
SYSTEM = """You are the incrementality analyst on a performance-marketing agency's account team, answering a skeptical CFO.
Data: Criteo Uplift v2.1, a randomized experiment with 13.98M users (85% eligible for ads, 15% held out), so
treated-vs-control gaps are causal. Targeting results are measured on a 30% hold-out the uplift model never saw.

Rules:
- Get every number from a tool call. Never estimate or compute figures yourself, except comparing two tool numbers.
- If a confidence interval includes zero, say the effect is not proven.
- Profit = incremental revenue - media spend. iROAS = incremental revenue / media spend. Cost is per EXPOSED user.
- Current sidebar assumptions: outcome={outcome}, ${value:,.2f} per incremental {outcome}, ${cost:,.2f} per exposed user,
  audience {audience:,} users, {conf:.0%} confidence. These are illustrative; say so when quoting dollars.
- Features f0-f11 are anonymized; never claim to know what they mean.
- Answer in under 170 words, plain English, lead with the decision-relevant number. Use $, ×, %, and pp units.
"""


def _secret(name: str):
    try:
        import streamlit as st
        return st.secrets.get(name) or os.getenv(name)
    except Exception:
        return os.getenv(name)


def api_key():
    try:
        import streamlit as st
        user = st.session_state.get("gemini_key")
    except Exception:
        user = None
    return user or _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")


def model_name() -> str:
    return _secret("GEMINI_MODEL") or DEFAULT_MODEL


def _log(msg: str) -> None:
    print(f"[analyst] {msg}", file=sys.stderr, flush=True)  # shows in Streamlit Cloud logs


def _declarations(types):
    decls = []
    for t in TOOLS:
        kw = {"name": t["name"], "description": t["description"]}
        if t["parameters"] and t["parameters"].get("properties"):
            kw["parameters_json_schema"] = t["parameters"]
        decls.append(types.FunctionDeclaration(**kw))
    return [types.Tool(function_declarations=decls)]


def _pick_model(client, failed: str, error: str, tried: set | None = None) -> str:
    """Pick a replacement model: the one Google's error suggests, else the newest untried 'flash' model,
    else (last resort) the newest untried 'flash-lite' model. Returns `failed` if nothing is left."""
    skip = set(tried or ()) | {failed}
    suggested = [m for m in re.findall(r"models/([\w.\-]+)", error) if m not in skip]
    if suggested:
        return suggested[0]
    try:
        names = [m.name.removeprefix("models/") for m in client.models.list()]
    except Exception:
        return failed
    bad = ("image", "tts", "audio", "live", "preview", "exp", "embedding")
    flash = sorted(n for n in names if "flash" in n and "lite" not in n and n not in skip and not any(x in n for x in bad))
    lite = sorted(n for n in names if "flash-lite" in n and n not in skip and not any(x in n for x in bad))
    return flash[-1] if flash else lite[-1] if lite else failed


def ask(question: str, history: list, toolbox: Toolbox, key: str, on_step=None) -> dict:
    """Gemini function-calling loop. Returns {'text', 'calls': [(name, args, output)], 'grounding', 'model'}."""
    from google import genai
    from google.genai import types

    step = on_step or (lambda m: None)
    a = toolbox.a
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=40_000))

    def make_config(thinking: bool):
        kw = dict(system_instruction=SYSTEM.format(outcome=a.outcome, value=a.value, cost=a.cost,
                                                   audience=a.audience, conf=a.conf),
                  tools=_declarations(types), temperature=0.2,
                  automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        if thinking:
            # Low thinking keeps answers fast: our tools do all the maths, Gemini only picks tools and writes.
            kw["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
        return types.GenerateContentConfig(**kw)

    config = make_config(thinking=True)
    contents = [types.Content(role="model" if h["role"] == "assistant" else "user",
                              parts=[types.Part.from_text(text=h["content"])]) for h in history]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    model, calls = model_name(), []
    tried = {model}
    for attempt in range(7):
        step(f"Asking {model}…" if attempt == 0 else "Writing the answer from the data…")
        t0 = time.time()
        resp = None
        for tries in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                break
            except Exception as e:
                msg = str(e)
                _log(f"request {attempt} try {tries} failed after {time.time() - t0:.1f}s: {msg[:200]}")
                if "404" in msg or "not found" in msg.lower():          # retired model -> suggested one
                    model = _pick_model(client, model, msg)
                elif "thinking" in msg.lower() and "400" in msg:       # model without thinking levels
                    config = make_config(thinking=False)
                elif "504" in msg or "503" in msg or "DEADLINE" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                    alt = _pick_model(client, model, "", tried)      # busy model: switch to an untried flash model
                    if alt != model:
                        step(f"{model} is busy right now. Switching to {alt}…")
                        model = alt
                        tried.add(alt)
                    elif tries == 0:
                        step("Google was slow to respond. Retrying once…")
                    else:
                        raise
                else:
                    raise
                if tries == 2:
                    raise
        fcs = resp.function_calls or []
        _log(f"request {attempt} ok in {time.time() - t0:.1f}s, tool calls: {[f.name for f in fcs]}")
        if not fcs:
            break
        contents.append(resp.candidates[0].content)
        parts = []
        for fc in fcs:
            args = dict(fc.args or {})
            step(f"Running `{fc.name}` on the experiment data…")
            try:
                out = toolbox.run(fc.name, args)
            except Exception as e:  # surface tool errors to the model rather than crash
                out = {"error": str(e)}
            calls.append((fc.name, args, out))
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
        contents.append(types.Content(role="user", parts=parts))

    text = (resp.text or "").strip()
    ctx = [a.value, a.cost, a.audience, a.conf * 100, 95, 1.0, 13.98, 85, 15, 70, 30, 100, 10, 20, 50]
    for _, inp, _ in calls:
        _flatten(inp, ctx)
    return {"text": text, "calls": calls, "model": model,
            "grounding": grounding_check(text, [c[2] for c in calls], ctx)}


# --------------------------------------------------------------------------- no-key brief
def deterministic_brief(toolbox: Toolbox) -> str:
    """Template narration from the same tool outputs: always available, fully grounded."""
    from ui import money, num
    inc = toolbox.run("get_incrementality", {})
    pol = toolbox.run("get_targeting_policy", {})
    strat = toolbox.run("compare_strategies", {})["strategies"]
    o = inc["assumptions"]["outcome"]
    best = max(strat, key=lambda s: s["Profit"] or -1e18)
    return "\n\n".join([
        f"**Do the ads work?** Yes: {o} rate {inc['control_rate'] * 100:.3f}% → {inc['treated_rate'] * 100:.3f}% "
        f"(+{inc['lift'] * 100:.3f} pp, p {S.fmt_p(inc['p_value'])}).",
        f"**Is attribution honest?** No: an exposure report would claim {num(inc['claimed_by_exposure_report'])} {o}s, "
        f"but only {num(inc['truly_incremental'])} ({inc['share_truly_incremental'] * 100:.0f}%) were caused by the ads.",
        f"**Who to target:** profit peaks at the top {pol['optimum_top_pct']}% by predicted uplift "
        f"({money(pol['optimum_profit'])} vs {money(pol['target_everyone_profit'])} for everyone); "
        f"validated out-of-sample gain {money(pol['validated_gain_vs_everyone'])}.",
        f"**Best strategy:** {best['Strategy']}: {money(best['Profit'])} profit at {best['iROAS']:.2f}× iROAS "
        f"on {money(best['Spend'])} of spend.",
    ])
