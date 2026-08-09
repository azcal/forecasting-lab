#!/usr/bin/env python3
"""
Merge a freshly pulled pipeline log into the dashboard's accumulated copy.

The refresh used to overwrite data/<sport>.csv with whatever that pipeline's
logs/predictions.csv held this morning. That is fine while a log only ever grows, and
wrong the moment one does not. A repo reset, a rotated log, a runner that rewrites its
frame with a narrower column set, a pipeline retired and its schedule switched off: any
of those silently shortened the dashboard's history, and there was no second copy of it
anywhere.

This merges instead.

  kept     rows already in data/<sport>.csv survive, always
  new      rows only in the pull are appended
  both     the freshly pulled version wins, because it is the one that may have had
           `result` filled in since the row was first logged
  columns  union, so a pipeline adding a column leaves older rows blank for it rather
           than dropping one side or the other

Row count is monotone by construction. The only way it falls is if the accumulated file
already contained duplicate keys, which this reports separately as a repair rather than
a loss.

Usage: python merge_logs.py <sport> <pulled.csv> <accumulated.csv>
"""
import os
import sys

import pandas as pd

# Row identity per pipeline, taken from what each runner actually writes. props keys on
# date plus athlete, matching the `date|athlete_id` key its runner dedupes on, because a
# player appears once per slate rather than once per season.
KEYS = {
    "wnba":   ["game_id"],
    "cs2":    ["match_id"],
    "soccer": ["id"],
    "nfl":    ["game_id"],
    "ncaaf":  ["game_id"],
    "nhl":    ["game_id"],
    "nba":    ["game_id"],
    "mma":    ["fight_id"],
    "mlb":    ["gamePk"],
    "props":  ["date", "athlete_id"],
    # Brasileirao, LigaMX and MLS share one log; `id` already encodes league,
    # date and both clubs, so it is unique across all three.
    "soccer_americas": ["id"],
}

# Column to sort the merged file on, so appended rows do not land at the bottom out of
# order. CS2 stamps its rows `ts` rather than `date`.
SORT_CANDIDATES = ("date", "ts")


def norm_key(s):
    """Canonical text form of a key column.

    `astype(str)` alone is not enough. A CSV column of ids reads back as float64 the moment
    it contains one blank, so a stored row keys on "2998938.0" while a freshly built row
    keys on "2998938". Those never match, the old row is never replaced, and every rerun
    adds a duplicate. Stripping the trailing .0 makes both sides agree.
    """
    return (s.astype(str).str.strip()
             .str.replace(r"\.0$", "", regex=True)
             .replace({"nan": "", "None": "", "<NA>": ""}))


def _keyframe(d, key):
    """Join the key columns into one string.

    Key dtypes drift between pulls: game_id reads as int64 one day and object the next,
    depending on whether anything non-numeric landed in the column. Comparing an int key
    to a string key never matches, so every row would duplicate and the file would double
    in size while looking like it merged. Both sides are cast before the join.
    """
    return d[key].apply(norm_key).agg("|".join, axis=1)


def freshness(d):
    """Latest date with a graded outcome, and how many rows are graded.

    Printed on both sides of every merge so a stale board is attributable in one look:
    if the PULLED file is already behind, the pipeline has not graded yet and nothing here
    can help; if the pulled file is current and the KEPT file is not, the merge is at
    fault. Without this the two look identical from the outside.
    """
    oc = next((c for c in ("result", "y", "actual") if c in d.columns), None)
    dc = next((c for c in ("date", "ts", "game_date") if c in d.columns), None)
    if oc is None or dc is None:
        return "no outcome column", 0
    g = d[d[oc].notna()]
    if not len(g):
        return "none graded", 0
    t = g[dc].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    ymd = t.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns]")
    if ymd.any():
        parsed.loc[ymd] = pd.to_datetime(t[ymd], format="%Y%m%d", errors="coerce")
    if (~ymd).any():
        parsed.loc[~ymd] = pd.to_datetime(t[~ymd], format="mixed", errors="coerce",
                                          utc=True).dt.tz_localize(None)
    last = parsed.max()
    return ("unparseable dates" if pd.isna(last) else f"{last:%Y-%m-%d}"), len(g)


def merge(sport, pulled_path, kept_path):
    key = KEYS.get(sport)
    if key is None:
        raise SystemExit(f"{sport}: no merge key configured. Add one to KEYS rather than "
                         f"letting this guess and silently mangle history.")

    pulled = pd.read_csv(pulled_path, low_memory=False)
    if not os.path.exists(kept_path):
        pulled.to_csv(kept_path, index=False)
        print(f"  {sport:7} first pull, {len(pulled)} rows kept")
        return len(pulled)

    kept = pd.read_csv(kept_path, low_memory=False)
    for name, d in (("pulled", pulled), ("accumulated", kept)):
        missing = [c for c in key if c not in d.columns]
        if missing:
            raise SystemExit(
                f"{sport}: the {name} file has no {missing}. The schema changed; fix the "
                f"key in merge_logs.py. Refusing to merge on a guess and lose history.")

    kept["_k"] = _keyframe(kept, key)
    pulled["_k"] = _keyframe(pulled, key)

    n_kept_raw = len(kept)
    kept = kept.drop_duplicates("_k", keep="last")
    repaired = n_kept_raw - len(kept)

    overlap = kept._k.isin(set(pulled._k)).sum()
    added = int((~pulled._k.isin(set(kept._k))).sum())

    new_cols = [c for c in pulled.columns if c not in kept.columns and c != "_k"]
    gone_cols = [c for c in kept.columns if c not in pulled.columns and c != "_k"]

    out = (pd.concat([kept, pulled], ignore_index=True)
             .drop_duplicates("_k", keep="last")
             .drop(columns="_k"))

    sort_on = next((c for c in SORT_CANDIDATES if c in out.columns), None)
    if sort_on:
        out = out.sort_values(sort_on, kind="stable").reset_index(drop=True)

    if len(out) < len(kept):
        raise SystemExit(f"{sport}: merge produced {len(out)} rows from {len(kept)} kept. "
                         f"That should be impossible; not writing.")

    out.to_csv(kept_path, index=False)
    bits = [f"{len(out)} rows", f"+{added} new", f"{overlap} refreshed"]
    if repaired:
        bits.append(f"{repaired} duplicate rows repaired")
    if new_cols:
        bits.append(f"columns added {new_cols}")
    if gone_cols:
        bits.append(f"no longer written {gone_cols}")
    pf, pn = freshness(pulled)
    kf, kn = freshness(out)
    bits.append(f"pulled graded to {pf} ({pn})")
    bits.append(f"now graded to {kf} ({kn})")
    print(f"  {sport:7} " + ", ".join(bits))
    if pf != kf:
        print(f"  {'':7} the pulled file and the merged file disagree on how current they "
              f"are. That points at the merge key, not the pipeline.")
    return len(out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
