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
    "CS2":    {"file": "data/cs2.csv",    "url": ""},
    "Soccer": {"file": "data/soccer.csv", "url": ""},
    "NFL":    {"file": "data/nfl.csv",    "url": ""},
    "NCAAF":  {"file": "data/ncaaf.csv",  "url": ""},
    "NHL":    {"file": "data/nhl.csv",    "url": ""},
    "NBA":    {"file": "data/nba.csv",    "url": ""},
    "MMA":    {"file": "data/mma.csv",    "url": ""},
}

STATUS = {
    "WNBA": "live", "CS2": "live",
    "Soccer": "live (season opens mid-Aug)",
    "NFL": "pre-season (Sept)", "NCAAF": "pre-season (late Aug)",
    "NHL": "pre-season (Oct)", "NBA": "pre-season (Oct)", "MMA": "live",
}

# which side the base-rate strategy always backs, per board listing convention
# Which side the base-rate strategy always backs. Keyed by board, or by (board, head)
# where a board runs more than one market and they have different baselines.
BASE_SIDE = {"CS2": "team 1", "MMA": "the alphabetically first fighter",
             ("MMA", "goes the distance"): "the distance",
             ("NFL", "spread, home -3.5"): "the away side at +3.5",
             ("NBA", "spread, home -3.5"): "the away side at +3.5"}


def base_side(board, head=None):
    return (BASE_SIDE.get((board, head)) or BASE_SIDE.get(board) or "the home side")

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

**Extra calls right per 100** is the plain-language version: how many more games the model gets
on the right side of than someone who backs the home side (team 1 on the CS2 board) every single
time, without ever looking at who is playing. It ignores confidence entirely, which is why it is
the easy number rather than the real one.

That comparison only means something if the model actually disagrees with the home side
sometimes, so each board reports how often it does and how it fares when it does. Across these
boards it backs against the home side on roughly a third of games and wins around 60% of them.

**It is not a comparison against a bookmaker.** No odds, lines or market prices enter any model
here or any number on this page. Everything is built from league APIs and open data and graded
against a base rate and an Elo floor. Nothing on this dashboard claims to beat a book, because
nothing here has ever seen one.

**The two floors.** *Base rate* is what you would score knowing nothing except how often the
home side wins in this league. Any model has to beat it to be worth anything. *Elo baseline*
is a simple power rating built from wins and losses alone. The gap between the Elo line and
the model line is the only part that is modelling work rather than a rating anyone could
build, and it is the number to look at hardest.

**Skill score** is the share of the base rate's error the model removes.

| skill score | reading |
|---|---|
| 15% or more | elite |
| 8% to 15% | strong |
| 4% to 8% | solid |
| 1.5% to 4% | thin |
| under 1.5% | no measurable edge |

Those cutoffs are mine, not an industry standard. There is no published scale for cross-sport
forecast skill. For a feel: every board here reading *strong* or better calls between 8 and 13
more games right per 100 than always backing the favourite, the one reading *thin* calls about
2 more, and the one with no measurable edge calls slightly fewer.

**The two numbers can disagree, and that is the point.** Extra calls counts direction only. Log
loss also scores confidence, so a model that is right 60% of the time while insisting it is
right 90% of the time gets punished by log loss and looks fine on a simple count. The clearest
example on this dashboard is the retired MLB first-five-innings board: fractionally positive on
log loss and fractionally negative on raw calls, which is what a board with no edge in either
direction looks like. It is on the Retired tab with the full workings.

**One caution.** The skill scores do not compare between sports the way they look. First-five-innings
baseball is close to a coin flip no matter who is modelling it, so there is far less to
predict there than in college football. A small number on a hard board is not a bad model.
"""

BOARD_NOTES = {
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
    "MMA": (
        "**New board, shipped 2026-07-28.** Built from Wikipedia bout templates across UFC, "
        "Bellator, PFL and ONE: 10,216 decided fights with method, round and finish time. "
        "Non-UFC bouts update fighter ratings but are never graded, since cross-promotion "
        "fights are rare enough that a shared rating scale across four organisations would "
        "rest on weakly connected pools. Fighters are ordered alphabetically, so the base "
        "rate is 0.500 and nothing is inherited from a listing convention. A third market, "
        "winner by method, is boarded but not shown here: it is a three-way outcome and this "
        "page grades binary markets. Distance is derived from the method model rather than "
        "fitted separately, so the two can never contradict each other. Why it reads thinner "
        "than the team sports: the Elo floor is worth +0.008 here against +0.113 in NCAAF, "
        "because across 10,216 bouts there are 4,534 distinct fighters, the average fighter "
        "appears about four and a half times in the whole dataset, and any fight can end "
        "with one punch."
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

Four targets were built, evaluated, and shut off. Each was tested once on a frozen holdout
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

#### MLB F5: first-five-innings winner
*Retired 2026-07-27.* The board never separated itself from a constant. Final live sample,
n = 1,300.

| metric | value |
|---|---|
| log loss | 0.6911 |
| base-rate log loss | 0.6917 |
| edge over base rate | +0.0006 |
| paired test, one-sided | t = 0.13, p = 0.45 |
| 95% interval on the edge | −0.0084 to +0.0096 |
| Elo floor | 0.6918, i.e. *above* the base rate |
| AUC | 0.5437 |
| hit rate | 52.8% |

Team strength carries no information about who leads after five innings: the Elo floor scored
worse than a constant. Two fixes were built and measured before the decision. A temperature
layer corrected a raw head that was overconfident by a factor of about 1.7 and was worth
roughly +0.003. A lineup layer using confirmed batting orders, prior-season batting lines and
platoon handedness, joined to all 19,474 games, was worth +0.0013 pooled across both holdouts
(t = 1.61) with the platoon component measuring actively negative and excluded.

Best achievable was roughly +0.006 against a bar of +0.010 with p < 0.01, pre-registered on
2026-07-26 before either fix was tested. The bar was not met and the board was retired the
following day.

Worth recording why the lineup layer was so much smaller here than on the basketball boards,
where the same feature construction is worth +0.017: MLB lineups churn plenty, a mean of 2.35
regulars missing and 3 or more absent in 42.7% of games. The limit is that one hitter is a
ninth of a lineup and gets about two plate appearances in five innings, where an NBA starter is
a quarter of the offence and plays three quarters of the game.

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
totals, an NCAAF lineup layer), and two live targets were retired after their reads came back
indistinguishable from the base rate (CS2 series length, MLB first-five-innings). The second was
retired against a bar committed to the repository before its candidate fixes were tested, which
is the point of registering one in advance. Early-season forecasts on
every board carry a calibration gate: predictions publish from day one, but the board withholds
actionability guidance until each league passes a games-played threshold (NFL Week 5, NHL and
NBA every team at 9 and 8 games played).
"""


def _spread_line(d):
    """The model's own spread, e.g. 'PIT -6.5'. Negative margin means the road side lays."""
    if "sp_margin" not in d.columns:
        return pd.Series([""] * len(d), index=d.index)
    m = pd.to_numeric(d.sp_margin, errors="coerce")
    half = (m.abs() * 2).round() / 2
    half = half.where(half % 1 != 0, half + 0.5)      # never quote a whole number
    lays = np.where(m >= 0, d.home, d.away)
    return pd.Series([f"{t} -{h:.1f}" if h == h else "" for t, h in zip(lays, half)],
                     index=d.index)


def outcome(sr):
    """Normalise a logged result to 0/1 with pushes and unplayed games as NaN.

    Pipelines write one of three conventions and the dashboard has to read all of them:
      0/1            already binary (CS2, WNBA, NHL, NBA, MMA)
      0/1/-1         binary with -1 marking a push (MLB, Soccer)
      signed margin  home score minus away score (NFL, NCAAF)
    Assuming binary silently discarded every margin that was not 0 or 1, which is why the
    NFL and NCAAF boards went blank the moment the dashboard started reading their real
    logs instead of hand-made snapshots.
    """
    v = pd.to_numeric(sr, errors="coerce")
    seen = set(v.dropna().unique())
    if seen <= {0.0, 1.0, -1.0}:
        return v.where(v.isin([0.0, 1.0]))
    return pd.Series(np.where(v > 0, 1.0, np.where(v < 0, 0.0, np.nan)),
                     index=v.index if hasattr(v, "index") else None)


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
    elif name == "CS2":
        # Series-length head retired 2026-07-26; see the Retired tab. Historical
        # p_3maps / maps3 columns may still be present in the log and are ignored.
        out.append(pd.DataFrame({"date": pd.to_datetime(d.ts).dt.tz_localize(None),
            "matchup": d.team1 + " vs " + d.team2, "p_model": d.p_team1,
            "p_floor": d.p_elo, "y": outcome(d.get("y")), "head": "series winner",
            "p_team": d.team1, "opp_team": d.team2}))
    elif name == "MMA":
        base = pd.DataFrame({"date": pd.to_datetime(d.date),
                             "matchup": d.A + " vs " + d.B})
        # Rows logged before the runner recorded the Elo floor have no p_elo. Fall back to
        # NaN rather than raising, which turns the floor line off instead of the board.
        floor = d["p_elo"] if "p_elo" in d.columns else np.nan
        out.append(base.assign(p_model=d.p_win_a, p_floor=floor, y=outcome(d.get("result")),
                               head="fight winner", p_team=d.A, opp_team=d.B))
        out.append(base.assign(p_model=d.p_distance, p_floor=np.nan, y=outcome(d.get("went_dist")),
                               head="goes the distance",
                               p_team="the distance", opp_team="a finish"))
    elif name == "Soccer":
        y = outcome(d.get("result"))
        out.append(pd.DataFrame({"date": pd.to_datetime(d.date),
            "matchup": d.home + " vs " + d.away + " (" + d.league + ")",
            "p_model": d.p_home, "p_floor": d.p_elo, "y": y,
            "head": "match winner (draws excluded)",
            "p_team": d.home, "opp_team": d.away}))
    else:
        heads = {"NFL": "game winner (ties push)", "NCAAF": "game winner (FBS vs FBS)",
                 "NHL": "game winner (incl OT/SO)", "NBA": "game winner"}
        y = outcome(d.get("result"))
        # Spread head. The runners log P(home covers) at fixed margin thresholds, so it
        # grades without needing to know what line a book actually posted. Only the +3.5
        # threshold is shown: near zero it duplicates the moneyline, and the other logged
        # thresholds are there for analysis rather than the page.
        if "p_sp_p35" in d.columns and d.p_sp_p35.notna().any():
            # Grade against the logged final margin. Falling back to `result` only works
            # on boards that happen to log a margin there, which is why NBA came up blank.
            if "final_margin" in d.columns:
                marg = pd.to_numeric(d.final_margin, errors="coerce")
            else:
                marg = pd.to_numeric(d.get("result"), errors="coerce")
                if set(marg.dropna().unique()) <= {0.0, 1.0, -1.0}:
                    marg = None
            if marg is not None and marg.notna().any():
                out.append(pd.DataFrame({
                    "date": pd.to_datetime(d.date),
                    "matchup": d.away + " @ " + d.home,
                    "p_model": d.p_sp_p35, "p_floor": np.nan,
                    "y": np.where(marg.isna(), np.nan, (marg > 3.5).astype(float)),
                    "head": "spread, home -3.5",
                    "p_team": d.home + " -3.5", "opp_team": d.away + " +3.5",
                    # The graded series is a fixed line, because a ladder cannot be scored
                    # as one number. What gets displayed is the model's own line, rounded
                    # to the nearest half point, which is the thing you actually shop.
                    "line_txt": _spread_line(d)}))
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
    # the plain-count lens: always backing the more likely side gets you max(base, 1-base)
    s["base_hit"] = max(base, 1 - base)
    s["extra"] = s["hit"] - s["base_hit"]
    # how often the model overrules that strategy, and how it does when it does
    contra = pm < 0.5
    s["contra_share"] = float(contra.mean())
    s["contra_hit"] = float(np.mean(y[contra] == 0)) if contra.sum() else float("nan")
    if np.isfinite(pf).all():
        s["ll_floor"] = ll(y, pf)
        s["elo_edge"] = s["ll_base"] - s["ll_floor"]
        s["layer_edge"] = s["ll_floor"] - s["ll_model"]
        s["layer_t"], s["layer_p"] = paired_t(y, pm, pf)
    else:
        # No rating floor exists for this market, so the whole edge is the model and the
        # meaningful test is against the base rate rather than against a floor.
        s["ll_floor"] = None
        s["elo_edge"] = 0.0
        s["layer_edge"] = s["ll_base"] - s["ll_model"]
        s["layer_t"], s["layer_p"] = paired_t(y, pm, np.full(len(y), base))
    return s


def verdict(board, s, head=None):
    side = base_side(board, head)
    bits = [f"Across **{s['n']:,}** graded forecasts this board calls "
            f"**{s['extra']*100:+.1f} games per 100** right that someone backing {side} every "
            f"single time would get wrong, and removes **{s['skill']:.1%}** of the error a "
            f"no-information guess would make, which reads as *{s['rating']}*."]
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


def metrics_block(board, gg, head=None):
    s = board_stats(gg)
    cols = st.columns(5)
    cols[0].metric("Graded forecasts", f"{s['n']:,}")
    cols[1].metric("Skill score", f"{s['skill']:.1%}", s["rating"], delta_color="off")
    side = base_side(board, head)
    cols[2].metric("Extra calls right per 100", f"{s['extra']*100:+.1f}",
                   f"vs always backing {side}", delta_color="off")
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
    st.markdown(verdict(board, s, head))
    if not math.isnan(s["contra_hit"]):
        st.caption(f"It overrules that strategy on **{s['contra_share']:.0%}** of games, backing "
                   f"against {side}, and wins **{s['contra_hit']:.0%}** of those. Without that, "
                   f"the number above would only be measuring the head start {side} gets.")

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
        lab = "" if math.isnan(r["layer_t"]) else f"t = {r['layer_t']:.1f}"
        fig.add_annotation(x=tot, y=r["label"], text=lab, showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=11, color="#898781"))
    # Heading lives in Streamlit rather than the plotly title: a horizontal legend sits in
    # the top margin and the two collide there.
    fig.update_layout(barmode="stack", height=max(360, 42 * len(rows) + 80),
                      margin=dict(t=40, b=10, r=90),
                      xaxis_title="log loss removed from a no-information guess",
                      yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=1.04, yanchor="bottom",
                                  x=0, xanchor="left"))
    st.markdown("**Where each board's edge comes from**")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey is what a simple win-loss power rating gets you. Blue is what the model "
               "adds on top, and it is the only part that is modelling rather than bookkeeping. "
               "The t value next to each bar says whether that blue segment could be luck: "
               "above 2 means roughly a 1-in-40 chance it is, above 4 and luck is ruled out. "
               "Markets with no rating floor, such as goes-the-distance, are all blue by "
               "definition and their t is measured against the base rate instead.")


def rolling_chart(g):
    """Fixed 150-fight window with a noise band.

    The old rule, min(150, max(50, n//6)), gave the smallest window to the boards with the
    least data, which are exactly the ones that need the most smoothing. At a 50-game
    window the standard error of a rolling log loss estimate runs 65-87% of a typical
    board's whole edge, so the line crossing its floor means nothing. Fixed at 150 for
    comparability, with the band drawn so a reader can see which wiggles are real."""
    n = 150
    if len(g) < n + 30:
        st.info(f"Rolling form needs about {n + 30} graded forecasts and this board has "
                f"{len(g)}. The running total below is the better read until then.")
        return
    y = g.y.values
    base = float(np.mean(y))
    idx = range(n, len(g) + 1)
    frame = pd.DataFrame({
        "model": [ll(y[i-n:i], g.p_model.values[i-n:i]) for i in idx],
        "Elo baseline": [ll(y[i-n:i], g.p_floor.values[i-n:i])
                         if np.isfinite(g.p_floor.values).all() else np.nan for i in idx],
        "base rate": [ll(y[i-n:i], np.full(n, base)) for i in idx]},
        index=g.date.values[n-1:])
    # 95% band around the base rate: anything inside it is indistinguishable from noise
    se = float(np.std(ll_vec(y, np.full(len(y), base)), ddof=1)) / math.sqrt(n)
    fig = go.Figure()
    fig.add_scatter(x=list(frame.index) + list(frame.index)[::-1],
                    y=list(frame["base rate"] - 1.96*se) + list(frame["base rate"] + 1.96*se)[::-1],
                    fill="toself", fillcolor="rgba(209,213,219,0.25)",
                    line=dict(width=0), hoverinfo="skip", name="noise band")
    for c, col in (("model", "#2563eb"), ("Elo baseline", "#9ca3af"), ("base rate", "#d1d5db")):
        fig.add_scatter(x=frame.index, y=frame[c], name=c, line=dict(color=col))
    fig.update_layout(title=f"Recent form: rolling {n}-forecast log loss (lower is better)",
                      height=340, margin=dict(t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The blue line is the model and it should sit below both grey lines. The "
               "shaded band is where a model with no edge at all would wander by chance, so "
               "blue inside the band is not evidence of anything either way. Only sustained "
               "separation below it counts.")


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

frames = {b: load_board(b) for b in SOURCES}


def board_rank(b):
    """Rank a board by its strongest market so the tabs lead with the best work."""
    best = -9.0
    for _, g in frames[b].groupby("head"):
        gg = g.dropna(subset=["y"])
        if len(gg) < 30:
            continue
        best = max(best, skill(gg.y.values, gg.p_model.values, float(gg.y.mean())))
    return best


boards = sorted(SOURCES, key=board_rank, reverse=True)
tabs = st.tabs(["Overview"] + boards + ["Retired"])

with tabs[0]:
    rows, chart_rows = [], []
    # Walk the boards in tab order, and within a board put its strongest market first,
    # so the table, the tabs and the chart below all read the same way.
    for b in boards:
        f = frames[b]
        graded = []
        for h, g in f.groupby("head"):
            gg = g.dropna(subset=["y"])
            if len(gg) < 30:
                continue
            graded.append((h, gg, board_stats(gg)))
        for h, gg, s in sorted(graded, key=lambda x: -x[2]["skill"]):
            rows.append({"board": b, "status": STATUS.get(b, "live"), "target": h,
                         "graded n": s["n"], "skill score": f"{s['skill']:.1%}",
                         "reading": s["rating"],
                         "extra calls per 100": f"{s['extra']*100:+.1f}",
                         "hit rate": f"{s['hit']:.1%}", "log loss": round(s["ll_model"], 4),
                         "base-rate LL": round(s["ll_base"], 4)})
            multi = f["head"].nunique() > 1
            chart_rows.append(dict(s, label=f"{b}: {h}" if multi else b, board=b, head=h))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("How to read these numbers", expanded=True):
        st.markdown(HOW_TO_READ)
    if chart_rows:
        contribution_chart(chart_rows)
    st.markdown("Four further targets were built and shut off. They are listed under the "
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
            metrics_block(b, gg, h)
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
                col = [f"{t} {p:.0%}" for t, p in zip(fav, pct)]
                label = "model favours"
                if "line_txt" in sh.columns and sh.line_txt.astype(str).str.len().gt(0).any():
                    col = list(sh.line_txt.values)
                    label = "model's line"
                show = pd.DataFrame({
                    "date": pd.to_datetime(sh.date).dt.strftime("%Y-%m-%d").values,
                    "matchup": sh.matchup.values, label: col})
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.caption("The percentage is that named side's own chance of winning, so 50% "
                           "means the model sees a coin flip. Home teams are listed second on "
                           "the @ boards and first on the soccer board, which is why the side "
                           "is named rather than left implied.")

with tabs[-1]:
    st.markdown(RETIRED_MD)
