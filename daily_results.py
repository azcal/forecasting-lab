#!/usr/bin/env python3
"""
Yesterday's results, board by board and pick by pick.

Reads the same CSVs the dashboard reads, so it needs the refresh to have run first. It
writes nothing.

Two things are graded differently here than on the dashboard, on purpose.

Spreads are graded at the line the board actually published, `sp_line`, rather than the
dashboard's fixed -3.5 head. If the board said Team A -6 and Team A won by 5, that is a
miss, and a results post should say so. The -3.5 head exists so the dashboard can compare
seasons at a constant line; this is about what somebody could have bet last night.

Props are graded against `grade_line`, the artificial line 2.5 above the forecast. That
line is not a book number, so the hit rate on it is a diagnostic rather than a betting
record, and the detail shows forecast against actual so you can see the projection working
even when the over/under call missed.

    python daily_results.py                    yesterday
    python daily_results.py --date 2026-08-05  a specific day, for backfilling
"""
import argparse
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_dates(col):
    """WNBA writes the integer 20260508; everything else writes an ISO string.

    Pandas reads that integer as nanoseconds since epoch and silently returns 1970-01-01,
    which drops the whole board from the recap without erroring.
    """
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col.astype("Int64").astype(str), format="%Y%m%d",
                              errors="coerce")
    return pd.to_datetime(col, format="mixed", errors="coerce")


def read(stem, dcol, day):
    path = os.path.join(DATA, f"{stem}.csv")
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_csv(path)
    except Exception:
        return None
    if dcol not in d.columns:
        return None
    d = d[parse_dates(d[dcol]).dt.date == day]
    return d if len(d) else None


def money(d, home, away, pcol, ycol, extra=None, sep="@"):
    """A two-way board: pick the side above 50%, grade against a 0/1 outcome."""
    d = d[d[pcol].notna() & d[ycol].notna()]
    d = d[d[ycol].isin([0.0, 1.0])]
    if not len(d):
        return None, []
    p = d[pcol].astype(float).values
    y = d[ycol].astype(float).values
    pick_home = p >= .5
    conf = np.where(pick_home, p, 1 - p)
    hit = (pick_home == (y == 1))
    lines = []
    for i, (_, r) in enumerate(d.iterrows()):
        pick = r[home] if pick_home[i] else r[away]
        won = r[home] if y[i] == 1 else r[away]
        tag = "" if extra is None else f" _{r[extra]}_"
        lines.append(f"| {r[away]} {sep} {r[home]}{tag} | {pick} {conf[i]:.0%} | {won} won | "
                     f"{'HIT' if hit[i] else 'miss'} |")
    return (y, p), lines


def spread(d, pcol_unused=None):
    """Graded at the line the board published, not a fixed one."""
    need = {"sp_line", "final_margin", "sp_margin", "home", "away"}
    if not need.issubset(d.columns):
        return None, []
    d = d[d.sp_line.notna() & d.final_margin.notna() & d.sp_margin.notna()]
    if not len(d):
        return None, []
    line = d.sp_line.astype(float).values          # + means home favoured by that much
    marg = d.final_margin.astype(float).values     # actual home margin
    proj = d.sp_margin.astype(float).values
    home_side = proj > line                        # board leans home at its own line
    covered = marg > line
    push = np.isclose(marg, line)
    hit = (home_side == covered) & ~push
    lines = []
    for i, (_, r) in enumerate(d.iterrows()):
        team = r.home if home_side[i] else r.away
        shown = -abs(line[i]) if home_side[i] == (line[i] > 0) else abs(line[i])
        res = (f"{r.home} by {marg[i]:.0f}" if marg[i] > 0 else
               f"{r.away} by {abs(marg[i]):.0f}" if marg[i] < 0 else "tied")
        out = "push" if push[i] else ("HIT" if hit[i] else "miss")
        lines.append(f"| {r.away} @ {r.home} | {team} {shown:+.1f} | {res} | {out} |")
    keep = ~push
    if not keep.sum():
        return None, lines
    # Hit/miss only. The board publishes a line rather than a price, and the threshold
    # search puts P(cover) at roughly break-even on every row by construction, so there is
    # no per-game probability here worth scoring. Inventing one would make log loss and
    # Brier look meaningful when they would just be a restatement of the hit rate.
    return (hit[keep].astype(float), None), lines


def props(d):
    """Forecast against actual, plus the over/under call on the artificial line."""
    need = {"player", "matchup", "proj", "grade_line", "p_grade", "actual"}
    if not need.issubset(d.columns):
        return None, []
    d = d[d.actual.notna() & d.p_grade.notna()]
    if not len(d):
        return None, []
    p = d.p_grade.astype(float).values
    y = (d.actual.astype(float) > d.grade_line.astype(float)).astype(float).values
    over = p >= .5
    hit = (over == (y == 1))
    err = (d.actual.astype(float) - d.proj.astype(float)).values
    lines = []
    for i, (_, r) in enumerate(d.sort_values("player").iterrows()):
        j = d.index.get_loc(r.name)
        lines.append(f"| {r.player} | {r.matchup} | {r.proj:.1f} | {r.actual:.0f} | "
                     f"{err[j]:+.1f} | {r.grade_line:.1f} | "
                     f"{'over' if over[j] else 'under'} | "
                     f"{'HIT' if hit[j] else 'miss'} |")
    return (y, p), lines


BOARDS = [
    ("wnba",            "WNBA",            "date", "money",  ("home", "away", "p_home", "result", None)),
    ("nba",             "NBA spread",      "date", "spread", ()),
    ("nhl",             "NHL",             "date", "money",  ("home", "away", "p_home", "result", None)),
    ("nfl",             "NFL spread",      "date", "spread", ()),
    ("ncaaf",           "NCAAF",           "date", "money",  ("home", "away", "p_home", "result", None)),
    ("soccer",          "Soccer Big 5",    "date", "money",  ("home", "away", "p_home", "result", "league")),
    ("soccer_americas", "Soccer Americas", "date", "money",  ("home", "away", "p_home", "result", "league")),
    ("cs2",             "CS2",             "ts",   "money",  ("team1", "team2", "p_team1", "y", None, "vs")),
    ("mma",             "MMA",             "date", "money",  ("A", "B", "p_win_a", "result", None, "vs")),
    ("props",           "Player props",    "date", "props",  ()),
]
HEADERS = {"money": "| matchup | pick | result | |\n|---|---|---|---|",
           "spread": "| matchup | pick | final | |\n|---|---|---|---|",
           "props": ("| player | game | forecast | actual | error | line | call | |\n"
                     "|---|---|---:|---:|---:|---:|---|---|")}


def score(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    hits = int(((p >= .5).astype(float) == y).sum())
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    br = float(np.mean((p - y) ** 2))
    return hits, len(y), ll, br


def main(day, post=False):
    summary, detail, ys, ps = [], [], [], []
    for stem, label, dcol, kind, args in BOARDS:
        d = read(stem, dcol, day)
        if d is None:
            continue
        if kind == "money":
            got, lines = money(d, *args)
        elif kind == "spread":
            got, lines = spread(d)
        else:
            got, lines = props(d)
        if not lines:
            continue
        if got is not None:
            y, p = got
            if p is None:                   # spreads: hit/miss only, see spread()
                y = np.asarray(y, float)
                summary.append((label, len(y), int(y.sum()), int((1 - y).sum()),
                                y.mean(), None, None))
            else:
                hits, n, ll, br = score(y, p)
                summary.append((label, n, hits, n - hits, hits / n, ll, br))
                ys.append(np.asarray(y, float))
                ps.append(np.asarray(p, float))
        detail.append((label, kind, lines))

    out = [f"## MBM results, {day:%A %d %B %Y}", ""]
    if not summary and not detail:
        # "Nothing settled" is ambiguous and was wrong the first time it printed: the
        # boards had played, the logs just had not been graded yet when the refresh pulled.
        # So say how current each file actually is, which distinguishes a genuinely quiet
        # day from a refresh that ran too early.
        out += ["Nothing settled. Most recent graded row in each file:", "",
                "| board | last graded | rows |", "|---|---|---:|"]
        stale = False
        for stem, label, dcol, _k, _a in BOARDS:
            path = os.path.join(DATA, f"{stem}.csv")
            if not os.path.exists(path):
                out.append(f"| {label} | file missing | |")
                continue
            try:
                d = pd.read_csv(path)
            except Exception:
                out.append(f"| {label} | unreadable | |")
                continue
            oc = next((c for c in ("result", "y", "actual") if c in d.columns), None)
            g = d[d[oc].notna()] if oc else d.iloc[0:0]
            if not len(g):
                out.append(f"| {label} | never | 0 |")
                continue
            last = parse_dates(g[dcol]).max()
            out.append(f"| {label} | {last:%Y-%m-%d} | {len(g):,} |")
            if last.date() < day:
                stale = True
        out.append("")
        if stale:
            out.append("**Every board is graded only up to before the day requested.** "
                       "That is a refresh that ran ahead of the pipelines rather than a "
                       "quiet night. Check that refresh-dashboard-data ran after the "
                       "morning pipeline runs, and re-run this once it has.")
        txt = "\n".join(out) + "\n"
        print(txt)
        _emit(txt)
        if post:
            to_discord(txt)
        return

    out += ["### By board", "",
            "| board | graded | hit | miss | hit rate | log loss | Brier |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    th = tn = 0
    for label, n, h, m, hr, ll, br in summary:
        lls = f"{ll:.4f}" if ll is not None else "n/a"
        brs = f"{br:.4f}" if br is not None else "n/a"
        out.append(f"| {label} | {n} | {h} | {m} | {hr:.1%} | {lls} | {brs} |")
        th += h
        tn += n
    if tn:
        out.append(f"| **All boards** | **{tn}** | **{th}** | **{tn - th}** | "
                   f"**{th / tn:.1%}** | | |")
    out.append("")
    if ys:
        p = np.concatenate(ps)
        conf = np.maximum(p, 1 - p)
        out.append(f"{int((conf < .60).sum())} of {len(p)} two-way forecasts were games the "
                   f"model itself called under 60%, so close to a toss-up.")
    if any(ll is None for *_, ll, _ in summary):
        out.append("")
        out.append("*Spread boards show hit rate only. They publish a line rather than a "
                   "price, so there is no per-game probability to score.*")
    if tn and tn < 30:
        out.append("")
        out.append("*Small day. One result moves these a lot.*")
    out.append("")

    for label, kind, lines in detail:
        out += [f"### {label}", "", HEADERS[kind]] + lines + [""]

    txt = "\n".join(out) + "\n"
    print(txt)
    _emit(txt)
    if post:
        to_discord(txt)


def _emit(txt):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(txt)


def for_discord(txt):
    """Re-lay markdown tables as fixed-width text in code blocks.

    Discord renders neither markdown tables nor the alignment row, so pasting the summary
    straight through shows a wall of pipes. GitHub's step summary does render them, so the
    conversion happens here rather than upstream and both surfaces get the right thing.
    """
    out, tbl = [], []

    def flush():
        if not tbl:
            return
        rows = [[c.strip().replace("**", "") for c in r.strip().strip("|").split("|")]
                for r in tbl if not set(r.replace("|", "").replace(":", "").strip()) <= {"-"}]
        if not rows:
            tbl.clear()
            return
        w = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(len(rows[0]))]
        out.append("```")
        for j, r in enumerate(rows):
            out.append("  ".join((r[i] if i < len(r) else "").ljust(w[i])
                                 for i in range(len(w))).rstrip())
            if j == 0:
                out.append("  ".join("-" * x for x in w))
        out.append("```")
        tbl.clear()

    for line in txt.split("\n"):
        if line.startswith("|"):
            tbl.append(line)
            continue
        flush()
        out.append(line.replace("### ", "**").replace("## ", "**")
                   if line.startswith("#") else line)
    flush()
    # close the bold on any heading line we opened
    return "\n".join(l + "**" if l.startswith("**") and not l.endswith("**") else l
                      for l in out)


def to_discord(txt):
    """Post the recap, split into messages that fit Discord's 2,000-character limit.

    Splits on board sections first so a table is never cut mid-row, then on rows if a
    single section is still too long, which a busy props night will be. Markdown tables do
    not render in Discord, so the tables are re-laid as fixed-width text inside a code
    block, which does.
    """
    hook = os.environ.get("DISCORD_WEBHOOK", "").strip()
    if not hook:
        print("no DISCORD_WEBHOOK set, printed only")
        return
    txt = for_discord(txt)
    chunks, cur = [], ""
    for block in txt.split("\n**"):
        block = block if block.startswith("**") else "**" + block
        if len(block) <= 1900:
            pieces = [block]
        else:                                   # split a long table on its rows
            head, *rows = block.split("\n")
            pieces, buf = [], head
            for row in rows:
                if len(buf) + len(row) > 1850:
                    pieces.append(buf)
                    buf = head + "\n" + row
                else:
                    buf += "\n" + row
            pieces.append(buf)
        for pc in pieces:
            if len(cur) + len(pc) > 1900:
                chunks.append(cur)
                cur = pc
            else:
                cur = (cur + "\n\n" + pc) if cur else pc
    if cur:
        chunks.append(cur)
    sent = 0
    for i, c in enumerate(chunks):
        body = {"content": c, "allowed_mentions": {"parse": []}}
        try:
            r = requests.post(hook, json=body, timeout=30)
            if r.status_code >= 300:
                print(f"discord: chunk {i + 1} failed, HTTP {r.status_code} {r.text[:120]}")
            else:
                sent += 1
        except Exception as ex:
            print(f"discord: chunk {i + 1} failed, {type(ex).__name__}")
        time.sleep(1)                           # stay well inside the webhook rate limit
    print(f"discord: sent {sent} of {len(chunks)} message(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to yesterday")
    ap.add_argument("--discord", action="store_true", help="also post to DISCORD_WEBHOOK")
    a = ap.parse_args()
    main(dt.date.fromisoformat(a.date) if a.date else dt.date.today() - dt.timedelta(days=1),
         post=a.discord)
