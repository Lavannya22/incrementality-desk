"""Look & feel: CSS, KPI cards, Plotly defaults. Colors are tied to meaning and never re-assigned."""
import plotly.graph_objects as go
import streamlit as st

INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"
TREAT, CONTROL = "#2563EB", "#94A3B8"
GOOD, BAD, WARN = "#059669", "#DC2626", "#D97706"
SEG_COLORS = {"Persuadables": GOOD, "Sure things": TREAT, "Lost causes": CONTROL, "Predicted backfire": BAD}
SEG_BLURB = {
    "Persuadables": "Buy because of the ad. This is where every incremental dollar comes from.",
    "Sure things": "Would have bought anyway. Ads here are a gift, not a driver.",
    "Lost causes": "Won't buy either way. Spend here is pure waste.",
    "Predicted backfire": "Model predicts the ad makes them LESS likely to act. Checked against the hold-out below.",
}

CSS = f"""
<style>
.block-container {{padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px;}}
header[data-testid="stHeader"] {{background: transparent;}}
.hero {{background: linear-gradient(120deg,#0B1B3A 0%,#12306B 55%,#1D4ED8 100%); color:#fff; border-radius:18px;
        padding: 26px 32px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(15,23,42,.18);}}
.hero h1 {{margin:0; font-size: 1.9rem; font-weight: 750; letter-spacing:-.01em; color:#fff;}}
.hero p {{margin:.35rem 0 0; color:#C7D6F5; font-size: 1.0rem; max-width: 980px;}}
.hero .pill {{display:inline-block; background:rgba(255,255,255,.14); border-radius:999px; padding:3px 12px;
              font-size:.78rem; margin-right:8px; margin-top:12px; color:#E6EEFF;}}
.kpi {{background:#fff; border:1px solid {GRID}; border-radius:14px; padding:14px 18px; height:100%;
       box-shadow: 0 1px 2px rgba(15,23,42,.04);}}
.kpi .label {{font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}; font-weight:600;}}
.kpi .value {{font-size:1.85rem; font-weight:750; color:{INK}; line-height:1.15; margin-top:2px;}}
.kpi .sub {{font-size:.8rem; color:{MUTED}; margin-top:2px;}}
.kpi.good .value {{color:{GOOD};}} .kpi.bad .value {{color:{BAD};}} .kpi.blue .value {{color:{TREAT};}}
.callout {{background:#fff; border:1px solid {GRID}; border-left:5px solid {TREAT}; border-radius:10px;
           padding:14px 18px; margin: 8px 0 14px; color:{INK};}}
.callout.good {{border-left-color:{GOOD};}} .callout.bad {{border-left-color:{BAD};}} .callout.warn {{border-left-color:{WARN};}}
.callout b.t {{display:block; margin-bottom:2px;}}
.section-title {{font-size:1.25rem; font-weight:700; color:{INK}; margin: 8px 0 2px;}}
.section-sub {{color:{MUTED}; font-size:.92rem; margin-bottom:10px;}}
.badge {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:650;}}
.badge.good {{background:#D1FAE5; color:#065F46;}} .badge.bad {{background:#FEE2E2; color:#991B1B;}}
.badge.warn {{background:#FEF3C7; color:#92400E;}} .badge.blue {{background:#DBEAFE; color:#1E40AF;}}
div[data-testid="stTabs"] button[role="tab"] {{font-weight:600; font-size:.95rem;}}
section[data-testid="stSidebar"] {{background:#fff; border-right:1px solid {GRID};}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pills: list[str]):
    chips = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = ""):
    st.markdown(f'<div class="kpi {tone}"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def callout(title: str, body: str, tone: str = ""):
    st.markdown(f'<div class="callout {tone}"><b class="t">{title}</b>{body}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str) -> str:
    return f'<span class="badge {tone}">{text}</span>'


def style(fig: go.Figure, height: int = 380, legend: bool = True, ytitle: str | None = None, xtitle: str | None = None):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=84, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.97, yanchor="top"),
        hoverlabel=dict(bgcolor="#fff", font_size=13, bordercolor=GRID),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        showlegend=legend, hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED), title=xtitle, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="#CBD5E1", tickfont=dict(color=MUTED), title=ytitle)
    return fig


def show(fig: go.Figure):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def money(x: float, d: int = 0) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9:
        return f"{sign}${x / 1e9:,.2f}B"
    if x >= 1e6:
        return f"{sign}${x / 1e6:,.2f}M"
    if x >= 1e4:
        return f"{sign}${x / 1e3:,.1f}K"
    return f"{sign}${x:,.{d}f}"


def num(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if abs(x) >= 1e4:
        return f"{x / 1e3:,.1f}K"
    return f"{x:,.0f}"


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"
