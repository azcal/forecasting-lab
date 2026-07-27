"""MustBeMoose Forecasting Lab -- model monitoring dashboard.
Eight sports-forecasting pipelines, evaluated on log loss and calibration against
market-free baselines. Data auto-commits daily from GitHub Actions pipelines.
"""
import math
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

RATING = [(0.15, "elite"), (0.08, "strong"), (0.04, "solid"),
          (0.015, "thin"), (-99.0, "no measurable edge")]

def rate(s):
    for thr, word in RATING:
        if s >= thr:
            return word
    return "no measurable edge"


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def skill(y, p, base):
    """Share of the base-rate model's error that this model removes."""
    b = ll(y, np.full(len(y), base))
    return 0.0 if b <= 0 else 1 - ll(y, p) / b


def equiv_call(L):
    """The single confidence level q whose log loss equals L. A model scoring L is doing
    exactly as well as someone who calls every game q-to-(1-q) and is right that often.
    Bisection rather than scipy so the app keeps a four-package requirements file."""
    if L >= 0.6931:
        return 0.5
    lo, hi = 0.5 + 1e-9, 1 - 1e-9
    H = lambda q: -(q * math.log(q) + (1 - q) * math.log(1 - q))
    for _ in range(60):
        mid = (lo + hi) / 2
        if H(mid) > L:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_t(y, p_model, p_ref):
    """t-statistic and one-sided p for the per-game log loss difference. Normal
    approximation on the p value, which is fine at n in the hundreds."""
    d = ll_vec(y, p_ref) - ll_vec(y, p_model)
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    if se <= 0:
        return float("nan"), float("nan")
    t = d.mean() / se
    return t, 0.5 * math.erfc(t / math.sqrt(2.0))


HOW_TO_READ = """
#### The one-minute version

Every forecast on every board is scored, including the ones that were wrong, and scored
against two opponents it could plausibly lose to. Nothing is cherry-picked.

**Log loss** is the score. Lower is better. It grades a forecast on two things at once: was
it right, and how confident was it. Saying "50/50" to everything scores 0.693. Being
confident and right is rewarded; being confident and wrong is punished hard. That is
deliberate, because a forecast that hedges everything is useless.

**A log loss is easier to read as a coin.** A model scoring 0.545 is doing exactly as well as
someone who calls every single game a 77/23 shot and is right that often. One scoring 0.691
is calling them 53/47, which is barely off a coin flip. Each board shows its own equivalent.

**The two floors.** *Base rate* is what you would score knowing nothing except how often the
home side wins in this league. Any model has to beat it to be worth anything. *Elo baseline*
is a simple power rating built from wins and losses alone. The gap between the Elo line and
the model line is the only part that is modelling work rather than a rating anyone could
build, and it is the number to look at hardest.

**Skill score** is the share of the base rate's error the model removes.

| skill score | reading | same as calling every game |
|---|---|---|
| 15% or more | elite | 72 / 28 |
| 8% to 15% | strong | 67 / 33 |
| 4% to 8% | solid | 62 / 38 |
| 1.5% to 4% | thin | 57 / 43 |
| under 1.5% | no measurable edge | 50 / 50 |

Those cutoffs are mine, not an industry standard, and the right-hand column assumes an evenly
matched league. There is no published scale for cross-sport forecast skill.

**One caution.** These do not compare between sports the way they look. First-five-innings
baseball is close to a coin flip no matter who is modelling it, so there is far less to
predict there than in college football. A small number on a hard board is not a bad model.
"""

BOARD_NOTES = {
    "MLB F5": (
        "**Under review.** This board clears its base rate by roughly +0.0006 log loss across "
        "1,300 live forecasts (t = 0.13, p = 0.45), which is not distinguishable from zero. Its "
        "Elo floor sits *above* the base rate, meaning team strength carries no information "
        "about who leads after five innings. A temperature calibration layer was added "
        "2026-07-26 after the raw head measured as overconfident by a factor of about 1.7 on "
        "two out-of-sample splits. A retirement bar was pre-registered the same day and is "
        "checked on every pipeline run: at the end of the 2026 regular season the calibrated "
        "edge over base rate must reach +0.010 with p < 0.01 on the full live sample, or the "
        "board is retired and written up."
    ),
    "CS2": (
        "**v2 shipped 2026-07-26.** The rating now updates on per-map round margin rather than "
        "map counts. On the frozen 2026 holdout this moved log loss from 0.6255 to 0.6220 and "
        "AUC from 0.6924 to 0.6980, so it reorders matches rather than only rescaling them. "
        "Watch the gap between the model and the Elo baseline on this board: it is close to "
        "zero, so almost all the value here is the rating itself rather than the feature layer. "
        "Forecasts logged before that date came from v1 and are tagged as such, so the rolling "
        "chart mixes both until v2 fills a full window. The series-length head was retired the "
        "same day; see the Retired tab."
    ),
    "NHL": (
        "**v2 shipped 2026-07-26.** About 22% of NHL games go past sixty minutes, home teams win "
        "those at almost exactly 50%, and whether a game gets there is not predictable from team "
        "strength (AUC 0.4877 on the dev season). The winner head is now fit as a three-way "
        "regulation/OT decomposition and recombined as P(home) = P(home regulation win) + "
        "P(past regulation) x 0.50, which models the coin flip explicitly instead of letting a "
        "direct binary absorb it. Worth +0.0007 to +0.0014 log loss across the dev season and "
        "both frozen holdouts, moving them to 0.6613 and 0.6821. That roughly quarter of the "
        "slate carries an irreducible log loss near 0.693, which caps what this board can score."
    ),
}

RETIRED_MD = """
### Retired boards

Three targets were built, evaluated, and shut off. Each was tested once on a frozen holdout
and retired against a stated bar rather than on judgement after the fact. They are published
here because a model that beats nothing is worth more as a documented negative than as a
green light.

#### CS2: series length (over/under 2.5 maps)
*Retired 2026-07-26.* Logistic head on the absolute Elo gap, predicting whether a best-of-three
goes the distance. Frozen 2026 holdout, n = 623.

| metric | value |
|---|---|
| log loss | 0.6833 |
| base-rate log loss | 0.6842 |
| edge over base rate | +0.0009 |
| paired test, one-sided | t = 0.27, p = 0.39 |
| AUC | 0.5241 |

The edge is indistinguishable from zero and the AUC says the model barely orders matches
better than a coin flip. The structural problem is worse than the headline: predictions span
0.231 to 0.469 with a standard deviation of 0.042, against a base rate of 0.4334. The model's
ceiling sits below its own base rate, so it can never call a long series. That is a resolution
failure, and post-hoc calibration cannot repair it. Retired rather than left paused.

#### NHL: totals
*Failed validation, never shipped.* Reference goals total came in at MAE 1.881 and 1.831 across
the two holdouts against a constant's 1.883 and 1.833, so the model was worth about two
thousandths of a goal. It was benched pending a shot-quality input. That input was tested on
2026-07-26 using MoneyPuck expected goals joined to all 11,870 games. xG added roughly +0.0007
log loss to the winner head and did not come close to rescuing totals, so the board stays
retired and the pipeline stays free of a licensed dependency.

#### NCAAF: lineup layer
*Failed validation, never shipped.* Player-availability features did not improve on the
team-state model out-of-sample.

---

**Why this tab exists.** Every board here is graded against two floors it could plausibly lose
to. Publishing only the survivors would make the floors decorative.
"""

METHOD_MD = """
**Method in one paragraph.** Each pipeline ingests public data (league APIs and open data),
builds strictly pregame features through a sequential state engine with no look-ahead, and
trains on a frozen historical window. Later seasons are untouched holdouts, evaluated once.
Live, every forecast on the full slate is graded on log loss against two floors: the base-rate
constant and an Elo baseline. A rolling skill monitor flags any target whose recent performance
drops below the base rate. Two candidate targets failed validation and never shipped (NHL
totals, an NCAAF lineup layer), and one live target was retired after its holdout read came
back indistinguishable from the base rate (CS2 series length). One live target carries a
pre-registered retirement bar with a dated decision point (MLB F5). Early-season forecasts on
every board carry a calibration gate: predictions publish from day one, but the board withholds
actionability guidance until each league passes a games-played threshold (NFL Week 5, NHL and
NBA every team at 9 and 8 games played).
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
            "p_floor": d.p_home_elo, "y": d.get("result"), "head": "game winner",
            "p_team": d.home, "opp_team": d.away}))
    elif name == "MLB F5":
        y = d.get("result")
        y = y.where(y.isin([0.0, 1.0]))          # ties are pushes
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date, format="%Y%m%d"),
            "matchup": d.away + " @ " + d.home, "p_model": d.p_home,
            "p_floor": d.p_home_elo, "y": y, "head": "first-5-innings winner",
            "p_team": d.home, "opp_team": d.away}))
    elif name == "CS2":
        # Series-length head retired 2026-07-26; see the Retired tab. Historical
        # p_3maps / maps3 columns may still be present in the log and are ignored.
        out.append(pd.DataFrame({"date": pd.to_datetime(d.ts).dt.tz_localize(None),
            "matchup": d.team1 + " vs " + d.team2, "p_model": d.p_team1,
            "p_floor": d.p_elo, "y": d.get("y"), "head": "series winner",
            "p_team": d.team1, "opp_team": d.team2}))
    elif name == "Soccer":
        y = d.get("result"); y = y.where(y.isin([0.0, 1.0]))
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date),
            "matchup": d.home + " vs " + d.away + " (" + d.league + ")",
            "p_model": d.p_home, "p_floor": d.p_elo, "y": y,
            "head": "match winner (draws excluded)",
            "p_team": d.home, "opp_team": d.away}))
    else:
        heads = {"NFL": "game winner (ties push)", "NCAAF": "game winner (FBS vs FBS)",
                 "NHL": "game winner (incl OT/SO)", "NBA": "game winner"}
        y = d.get("result"); y = y.where(y.isin([0.0, 1.0]))
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date),
            "matchup": d.away + " @ " + d.home, "p_model": d.p_home,
            "p_floor": d.p_elo, "y": y, "head": heads[name],
            "p_team": d.home, "opp_team": d.away}))
    f = pd.concat(out, ignore_index=True).sort_values("date").reset_index(drop=True)
    f["board"] = name
    return f


def board_stats(gg):
    """Everything the cards, the verdict line and the contribution chart need."""
    y = gg.y.values.astype(float)
    pm = gg.p_model.values.astype(float)
    pf = gg.p_floor.values.astype(float)
    base = float(np.mean(y))
    s = {"n": len(y), "base": base, "ll_model": ll(y, pm),
         "ll_base": ll(y, np.full(len(y), base)),
         "hit": float(np.mean((pm > .5) == (y == 1)))}
    s["skill"] = 1 - s["ll_model"] / s["ll_base"] if s["ll_base"] > 0 else 0.0
    s["rating"] = rate(round(s["skill"], 3))
    s["call"] = equiv_call(s["ll_model"])
    if np.isfinite(pf).all():
        s["ll_floor"] = ll(y, pf)
        s["elo_edge"] = s["ll_base"] - s["ll_floor"]
        s["layer_edge"] = s["ll_floor"] - s["ll_model"]
        s["layer_t"], s["layer_p"] = paired_t(y, pm, pf)
    else:
        s["ll_floor"] = None
        s["elo_edge"] = 0.0
        s["layer_edge"] = s["ll_base"] - s["ll_model"]
        s["layer_t"], s["layer_p"] = float("nan"), float("nan")
    return s


def verdict(board, s):
    q = s["call"] * 100
    bits = [f"Across **{s['n']:,}** graded forecasts this board removes **{s['skill']:.1%}** of "
            f"the error a no-information guess would make, which reads as *{s['rating']}*. "
            f"That is the same as calling every game a **{q:.0f}/{100-q:.0f}** shot and being "
            f"right that often."]
    if s["ll_floor"] is not None:
        share = s["layer_edge"] / max(s["elo_edge"] + s["layer_edge"], 1e-9)
        if not math.isnan(s["layer_t"]) and s["layer_t"] >= 2:
            conf = "and that gap is large enough to rule out luck"
        elif not math.isnan(s["layer_t"]) and s["layer_t"] >= 1:
            conf = "though the sample is not yet big enough to rule out luck"
        else:
            conf = "which is not distinguishable from luck"
        bits.append(f"**{share:.0%}** of that comes from the model layer on top of a simple "
                    f"power rating, {conf}.")
    return " ".join(bits)


def metrics_block(board, gg):
    s = board_stats(gg)
    q = s["call"] * 100
    cols = st.columns(5)
    cols[0].metric("Graded forecasts", f"{s['n']:,}")
    cols[1].metric("Skill score", f"{s['skill']:.1%}", s["rating"], delta_color="off")
    cols[2].metric("Same as calling every game", f"{q:.0f} / {100-q:.0f}")
    cols[3].metric("Hit rate", f"{s['hit']:.1%}")
    if s["ll_floor"] is not None:
        tt = "" if math.isnan(s["layer_t"]) else f"t = {s['layer_t']:.1f}"
        cols[4].metric("Model adds over a plain rating", f"{s['layer_edge']:+.4f}", tt,
                       delta_color="off")
        st.caption(f"Log loss: model {s['ll_model']:.4f} · Elo baseline {s['ll_floor']:.4f} · "
                   f"base rate {s['ll_base']:.4f}. Lower is better and a coin flip is 0.6931.")
    else:
        cols[4].metric("Model adds over a plain rating", "n/a")
        st.caption(f"Log loss: model {s['ll_model']:.4f} · base rate {s['ll_base']:.4f}. "
                   f"Lower is better and a coin flip is 0.6931.")
    st.markdown(verdict(board, s))

    w = gg.tail(300)
    if len(w) >= 100:
        edge = ll(w.y, np.full(len(w), s["base"])) - ll(w.y, w.p_model)
        ok = edge > 0
        st.markdown(("\U0001F7E2 **Model health: OK** " if ok else
                     "\U0001F534 **Model health: under review** ")
                    + f"(rolling skill vs base rate: {edge:+.4f} log loss over the "
                      f"last {len(w)} forecasts)")
    return s


def contribution_chart(stats):
    """Stacked bar: how much of each board's edge is the rating, how much is the model."""
    rows = sorted(stats, key=lambda r: -(r["elo_edge"] + r["layer_edge"]))
    names = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_bar(y=names, x=[r["elo_edge"] for r in rows], orientation="h",
                name="Elo rating", marker_color="#b4b2a9",
                hovertemplate="Elo rating: %{x:+.4f}<extra></extra>")
    fig.add_bar(y=names, x=[r["layer_edge"] for r in rows], orientation="h",
                name="model layer", marker_color="#2563eb",
                hovertemplate="model layer: %{x:+.4f}<extra></extra>")
    for r in rows:
        tot = r["elo_edge"] + r["layer_edge"]
        lab = "t n/a" if math.isnan(r["layer_t"]) else f"t = {r['layer_t']:.1f}"
        fig.add_annotation(x=tot, y=r["label"], text=lab, showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=11, color="#898781"))
    fig.update_layout(barmode="stack", height=max(360, 42 * len(rows) + 90),
                      margin=dict(t=44, b=10, r=90),
                      title="Where each board's edge comes from",
                      xaxis_title="log loss removed from a no-information guess",
                      yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey is what a simple win-loss power rating gets you. Blue is what the model "
               "adds on top, and it is the only part that is modelling rather than bookkeeping. "
               "The t value next to each bar says whether that blue segment could be luck: "
               "above 2 means roughly a 1-in-40 chance it is, above 4 and luck is ruled out.")


def rolling_chart(g):
    n = min(150, max(50, len(g) // 6))
    y = g.y.values
    base = float(np.mean(y))
    frame = pd.DataFrame({
        "model": [ll(y[max(0, i-n):i], g.p_model.values[max(0, i-n):i]) for i in range(n, len(g)+1)],
        "Elo baseline": [ll(y[max(0, i-n):i], g.p_floor.values[max(0, i-n):i])
                         if np.isfinite(g.p_floor.values).all() else np.nan for i in range(n, len(g)+1)],
        "base rate": [ll(y[max(0, i-n):i], np.full(min(n, i), base)) for i in range(n, len(g)+1)]},
        index=g.date.values[n-1:])
    fig = go.Figure()
    for c, col in (("model", "#2563eb"), ("Elo baseline", "#9ca3af"), ("base rate", "#d1d5db")):
        fig.add_scatter(x=frame.index, y=frame[c], name=c, line=dict(color=col))
    fig.update_layout(title=f"Recent form: rolling {n}-forecast log loss (lower is better)",
                      height=340, margin=dict(t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The blue line is the model and it should sit below both grey lines. Anywhere it "
               "crosses above them, the model was doing worse than guessing over that stretch.")


def calibration_chart(g):
    b = pd.cut(g.p_model, np.arange(0, 1.01, .1))
    cal = g.groupby(b, observed=True).agg(pred=("p_model", "mean"), act=("y", "mean"),
                                          n=("y", "size")).dropna()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                    line=dict(dash="dash", color="#d1d5db"))
    fig.add_scatter(x=cal.pred, y=cal.act, mode="markers+lines", name="model",
                    marker=dict(size=np.clip(cal.n / 8, 6, 26), color="#2563eb"))
    fig.update_layout(title="Honesty check: does a 70% forecast win 70% of the time?",
                      xaxis_title="what the model said", yaxis_title="what actually happened",
                      height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dots on the dashed line mean the stated confidence is accurate. Above the line "
               "the model is underselling itself, below it the model is overconfident. Bigger "
               "dots cover more forecasts, so read those hardest.")


def skill_chart(g):
    y, pm = g.y.values, np.clip(g.p_model.values, 1e-6, 1 - 1e-6)
    base = float(np.mean(y))
    pb = np.full_like(y, base, dtype=float)
    per = (-(y * np.log(pb) + (1 - y) * np.log(1 - pb))) - (-(y * np.log(pm) + (1 - y) * np.log(1 - pm)))
    fig = go.Figure()
    fig.add_scatter(x=g.date, y=np.cumsum(per), name="running total", line=dict(color="#16a34a"))
    fig.add_hline(y=0, line=dict(color="#d1d5db", dash="dash"))
    fig.update_layout(title="Running total: how far ahead of a no-information guess",
                      height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Every forecast nudges this line up if the model beat the base rate on that game "
               "and down if it did not. A steady climb is what real signal looks like. A line "
               "that wanders sideways or falls is luck, and the health monitor above watches "
               "for exactly that.")


st.title("MustBeMoose Forecasting Lab")
st.caption("Eight automated sports-forecasting pipelines. Frozen models, walk-forward "
           "validation, daily grading against market-free baselines via GitHub Actions. "
           "All results are out-of-sample, and retired boards are published alongside the "
           "live ones.")
st.markdown("Built by Mark Parsons, CPHR · [Code & methodology](https://github.com/azcal/forecasting-lab)")

boards = list(SOURCES)
tabs = st.tabs(["Overview"] + boards + ["Retired"])
frames = {b: load_board(b) for b in boards}

with tabs[0]:
    rows, chart_rows = [], []
    for b, f in frames.items():
        for h, g in f.groupby("head"):
            gg = g.dropna(subset=["y"])
            if len(gg) < 30:
                continue
            s = board_stats(gg)
            q = s["call"] * 100
            rows.append({"board": b, "status": STATUS.get(b, "live"), "target": h,
                         "graded n": s["n"], "skill score": f"{s['skill']:.1%}",
                         "reading": s["rating"], "same as calling": f"{q:.0f}/{100-q:.0f}",
                         "hit rate": f"{s['hit']:.1%}", "log loss": round(s["ll_model"], 4),
                         "base-rate LL": round(s["ll_base"], 4)})
            chart_rows.append(dict(s, label=b if b != "CS2" else "CS2 winner"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("How to read these numbers", expanded=True):
        st.markdown(HOW_TO_READ)
    if chart_rows:
        contribution_chart(chart_rows)
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
            metrics_block(b, gg)
            with st.expander("How to read these numbers"):
                st.markdown(HOW_TO_READ)
            if b in BOARD_NOTES:
                st.info(BOARD_NOTES[b])
            c1, c2 = st.columns(2)
            with c1:
                rolling_chart(gg)
                skill_chart(gg)
            with c2:
                calibration_chart(gg)
                up = g[g.y.isna()].tail(8)
                sh = (up if len(up) else g.tail(8)).copy()
                pm_ = sh.p_model.values.astype(float)
                fav = np.where(pm_ >= 0.5, sh.p_team.values, sh.opp_team.values)
                pct = np.where(pm_ >= 0.5, pm_, 1 - pm_)
                st.markdown("**Latest forecasts**")
                show = pd.DataFrame({
                    "date": pd.to_datetime(sh.date).dt.strftime("%Y-%m-%d").values,
                    "matchup": sh.matchup.values,
                    "model favours": [f"{t} {p:.0%}" for t, p in zip(fav, pct)]})
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.caption("The percentage is that named side's own chance of winning, so 50% "
                           "means the model sees a coin flip. Home teams are listed second on "
                           "the @ boards and first on the soccer board, which is why the side "
                           "is named rather than left implied.")

with tabs[-1]:
    st.markdown(RETIRED_MD)
