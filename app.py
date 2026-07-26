"""MustBeMoose Forecasting Lab -- model monitoring dashboard.
Eight sports-forecasting pipelines, evaluated on log loss / Brier / calibration
against market-free baselines. Data auto-commits daily from GitHub Actions pipelines.
"""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="MustBeMoose Forecasting Lab", layout="wide")

# When the pipeline repos are public, set raw URLs here and the app goes fully live.
SOURCES = {
    "WNBA":   {"file": "data/wnba.csv",   "url": ""},
    "MLB F5": {"file": "data/mlb.csv",    "url": ""},
    "CS2":    {"file": "data/cs2.csv",    "url": ""},
    "Soccer": {"file": "data/soccer.csv", "url": ""},
    "NFL":    {"file": "data/nfl.csv",    "url": ""},
    "NCAAF":  {"file": "data/ncaaf.csv",  "url": ""},
    "NHL":    {"file": "data/nhl.csv",    "url": ""},
    "NBA":    {"file": "data/nba.csv",    "url": ""},
}

STATUS = {
    "WNBA": "live", "MLB F5": "live (under review)", "CS2": "live",
    "Soccer": "live (season opens mid-Aug)",
    "NFL": "pre-season (Sept)", "NCAAF": "pre-season (late Aug)",
    "NHL": "pre-season (Oct)", "NBA": "pre-season (Oct)",
}

# Shown under the metric cards on each board tab.
BOARD_NOTES = {
    "MLB F5": (
        "**Under review.** This board clears its base rate by roughly +0.0006 log loss "
        "across 1,300 live forecasts (t = 0.13, p = 0.45), which is not distinguishable "
        "from zero. Its Elo floor sits *above* the base rate, meaning team strength "
        "carries no information about who leads after five innings. A temperature "
        "calibration layer was added 2026-07-26 after the raw head measured as "
        "overconfident by a factor of about 1.7 on two out-of-sample splits. A retirement "
        "bar was pre-registered the same day and is checked on every pipeline run: at the "
        "end of the 2026 regular season the calibrated edge over base rate must reach "
        "+0.010 with p < 0.01 on the full live sample, or the board is retired and "
        "written up."
    ),
    "CS2": (
        "**v2 shipped 2026-07-26.** The rating now updates on per-map round margin rather "
        "than map counts. On the frozen 2026 holdout this moved log loss from 0.6255 to "
        "0.6220 and AUC from 0.6924 to 0.6980, so it reorders matches rather than only "
        "rescaling them. Forecasts logged before that date came from v1 and are tagged as "
        "such in the pipeline log, so the rolling chart below mixes both until v2 fills a "
        "full window. The series-length head was retired the same day; see the Retired tab."
    ),
}

RETIRED_MD = """
### Retired boards

Three targets were built, evaluated, and shut off. Each was tested once on a frozen
holdout and retired against a stated bar rather than on judgement after the fact. They
are listed here because a model that beats nothing is worth more as a documented negative
than as a green light.

#### CS2: series length (over/under 2.5 maps)
*Retired 2026-07-26.* Logistic head on the absolute Elo gap, predicting whether a Bo3 goes
the distance. Frozen 2026 holdout, n = 623.

| metric | value |
|---|---|
| log loss | 0.6833 |
| base-rate log loss | 0.6842 |
| edge over base rate | +0.0009 |
| paired test, one-sided | t = 0.27, p = 0.39 |
| AUC | 0.5241 |
| resolution / uncertainty | 0.00675 / 0.24556 |

The edge is indistinguishable from zero and the AUC says the model barely orders matches
better than a coin flip. The structural problem is worse than the headline: predictions
span 0.231 to 0.469 with a standard deviation of 0.042, against a base rate of 0.4334.
The model's ceiling sits below its own base rate, so it can never call a long series. That
is a resolution failure, and post-hoc calibration cannot repair it. Retired rather than
left paused indefinitely.

#### NHL: totals
*Failed validation, never shipped.* Did not beat its base rate out-of-sample without a
shot-quality input the pipeline does not currently carry.

#### NCAAF: lineup layer
*Failed validation, never shipped.* Player-availability features did not improve on the
team-state model out-of-sample.

---

**Why this tab exists.** Every board on this dashboard is graded against two floors it
could plausibly lose to. Publishing only the survivors would make the floors decorative.
"""

METHOD_MD = """
**Method in one paragraph.** Each pipeline ingests public data (league APIs / open data),
builds strictly pregame features through a sequential state engine (no look-ahead), and
trains on a frozen historical window. Later seasons are untouched holdouts, evaluated once.
Live, every forecast on the full slate is graded on log loss and Brier score against two
floors: the base-rate constant and an Elo baseline. A rolling skill monitor flags any
target whose recent performance drops below the base rate. Two candidate targets failed
validation and never shipped (NHL totals, an NCAAF lineup layer), and one live target was
retired after its holdout read came back indistinguishable from the base rate (CS2 series
length). One live target carries a pre-registered retirement bar with a dated decision
point (MLB F5). Early-season forecasts on every board carry a calibration gate: predictions
publish from day one, but the board withholds actionability guidance until each league
passes a games-played threshold (e.g. NFL Week 5, NHL/NBA every team 9/8 GP).
"""


@st.cache_data(ttl=3600)
def load_board(name):
    cfg = SOURCES[name]
    src = cfg["url"] or cfg["file"]
    d = pd.read_csv(src)
    out = []
    if name == "WNBA":
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date, format="%Y%m%d"),
            "matchup": d.away + " @ " + d.home, "p_model": d.p_home,
            "p_floor": d.p_home_elo, "y": d.get("result"), "head": "game winner"}))
    elif name == "MLB F5":
        y = d.get("result")
        y = y.where(y.isin([0.0, 1.0]))          # ties are pushes
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date, format="%Y%m%d"),
            "matchup": d.away + " @ " + d.home, "p_model": d.p_home,
            "p_floor": d.p_home_elo, "y": y, "head": "first-5-innings winner"}))
    elif name == "CS2":
        # Series-length head retired 2026-07-26; see the Retired tab. Historical
        # p_3maps / maps3 columns may still be present in the log and are ignored.
        out.append(pd.DataFrame({"date": pd.to_datetime(d.ts).dt.tz_localize(None),
            "matchup": d.team1 + " vs " + d.team2, "p_model": d.p_team1,
            "p_floor": d.p_elo, "y": d.get("y"), "head": "series winner"}))
    elif name == "Soccer":
        y = d.get("result"); y = y.where(y.isin([0.0, 1.0]))
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date),
            "matchup": d.home + " vs " + d.away + " (" + d.league + ")",
            "p_model": d.p_home, "p_floor": d.p_elo, "y": y,
            "head": "match winner (draws excluded)"}))
    else:
        heads = {"NFL": "game winner (ties push)", "NCAAF": "game winner (FBS vs FBS)",
                 "NHL": "game winner (incl OT/SO)", "NBA": "game winner"}
        y = d.get("result"); y = y.where(y.isin([0.0, 1.0]))
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date),
            "matchup": d.away + " @ " + d.home, "p_model": d.p_home,
            "p_floor": d.p_elo, "y": y, "head": heads[name]}))
    f = pd.concat(out, ignore_index=True).sort_values("date").reset_index(drop=True)
    f["board"] = name
    return f


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def metrics_block(g):
    y, pm, pf = g.y.values, g.p_model.values, g.p_floor.values
    base = float(np.mean(y))
    cols = st.columns(5)
    cols[0].metric("Graded forecasts", f"{len(g):,}")
    cols[1].metric("Log loss (model)", f"{ll(y, pm):.4f}")
    fl = ll(y, pf) if np.isfinite(pf).all() else None
    cols[2].metric("Log loss (Elo baseline)", f"{fl:.4f}" if fl else "n/a")
    cols[3].metric("Log loss (base rate)", f"{ll(y, np.full_like(y, base)):.4f}")
    cols[4].metric("Hit rate", f"{np.mean((pm > .5) == (y == 1)):.1%}")
    w = g.tail(300)
    if len(w) >= 100:
        edge = ll(w.y, np.full(len(w), base)) - ll(w.y, w.p_model)
        ok = edge > 0
        st.markdown(("🟢 **Model health: OK** " if ok else "🔴 **Model health: under review** ")
                    + f"(rolling skill vs base rate: {edge:+.4f} log loss over last {len(w)} forecasts)")


def rolling_chart(g):
    n = min(150, max(50, len(g)//6))
    y = g.y.values
    base = float(np.mean(y))
    frame = pd.DataFrame({
        "model": [ll(y[max(0,i-n):i], g.p_model.values[max(0,i-n):i]) for i in range(n, len(g)+1)],
        "Elo baseline": [ll(y[max(0,i-n):i], g.p_floor.values[max(0,i-n):i])
                         if np.isfinite(g.p_floor.values).all() else np.nan for i in range(n, len(g)+1)],
        "base rate": [ll(y[max(0,i-n):i], np.full(min(n,i), base)) for i in range(n, len(g)+1)]},
        index=g.date.values[n-1:])
    fig = go.Figure()
    for c, col in (("model", "#2563eb"), ("Elo baseline", "#9ca3af"), ("base rate", "#d1d5db")):
        fig.add_scatter(x=frame.index, y=frame[c], name=c, line=dict(color=col))
    fig.update_layout(title=f"Rolling {n}-forecast log loss (lower = better)",
                      height=340, margin=dict(t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def calibration_chart(g):
    b = pd.cut(g.p_model, np.arange(0, 1.01, .1))
    cal = g.groupby(b, observed=True).agg(pred=("p_model", "mean"), act=("y", "mean"),
                                          n=("y", "size")).dropna()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                    line=dict(dash="dash", color="#d1d5db"))
    fig.add_scatter(x=cal.pred, y=cal.act, mode="markers+lines", name="model",
                    marker=dict(size=np.clip(cal.n/8, 6, 26), color="#2563eb"))
    fig.update_layout(title="Calibration: forecast probability vs observed frequency",
                      xaxis_title="forecast", yaxis_title="observed", height=340,
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def skill_chart(g):
    y, pm = g.y.values, np.clip(g.p_model.values, 1e-6, 1-1e-6)
    base = float(np.mean(y))
    pb = np.full_like(y, base, dtype=float)
    per = (-(y*np.log(pb) + (1-y)*np.log(1-pb))) - (-(y*np.log(pm) + (1-y)*np.log(1-pm)))
    fig = go.Figure()
    fig.add_scatter(x=g.date, y=np.cumsum(per), name="cumulative skill",
                    line=dict(color="#16a34a"))
    fig.add_hline(y=0, line=dict(color="#d1d5db", dash="dash"))
    fig.update_layout(title="Cumulative skill vs base-rate baseline (rising = sustained signal)",
                      height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


st.title("MustBeMoose Forecasting Lab")
st.caption("Eight automated sports-forecasting pipelines. Frozen models, walk-forward "
           "validation, daily grading against market-free baselines via GitHub Actions. "
           "All results below are out-of-sample. Retired boards are published alongside "
           "the live ones.")
st.markdown("Built by Mark Parsons, CPHR · [Code & methodology](https://github.com/azcal/forecasting-lab)")

boards = list(SOURCES)
tabs = st.tabs(["Overview"] + boards + ["Retired"])

frames = {b: load_board(b) for b in boards}

with tabs[0]:
    rows = []
    for b, f in frames.items():
        for h, g in f.groupby("head"):
            gg = g.dropna(subset=["y"])
            if len(gg) < 30:
                continue
            base = float(gg.y.mean())
            rows.append({"board": b, "status": STATUS.get(b, "live"), "target": h, "graded n": len(gg),
                         "log loss": round(ll(gg.y, gg.p_model), 4),
                         "base-rate LL": round(ll(gg.y, np.full(len(gg), base)), 4),
                         "hit rate": f"{np.mean((gg.p_model > .5) == (gg.y == 1)):.1%}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("Three further targets were built and shut off. They are listed under the "
                "**Retired** tab with the test that killed each one.")
    st.markdown(METHOD_MD)

for i, b in enumerate(boards, start=1):
    with tabs[i]:
        f = frames[b]
        for h, g in f.groupby("head"):
            gg = g.dropna(subset=["y"]).reset_index(drop=True)
            st.subheader(f"{b}: {h}")
            if len(gg) < 30:
                st.info("Awaiting graded forecasts.")
                continue
            metrics_block(gg)
            if b in BOARD_NOTES:
                st.info(BOARD_NOTES[b])
            c1, c2 = st.columns(2)
            with c1:
                rolling_chart(gg)
                skill_chart(gg)
            with c2:
                calibration_chart(gg)
                up = g[g.y.isna()].tail(8)
                st.markdown("**Latest forecasts**")
                show = (up if len(up) else g.tail(8))[["date", "matchup", "p_model"]]
                show = show.assign(date=pd.to_datetime(show.date).dt.strftime("%Y-%m-%d"),
                                   p_model=show.p_model.round(3))
                st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[-1]:
    st.markdown(RETIRED_MD)
