"""
Send the published NFL picks to the Model Collective.

Reads data/picks.csv and nothing else in this repo. Writes nothing, changes nothing,
touches no other pipeline. Standard library only, so it needs no requirements file.

Usage:
    python collective_submit.py            # dry run, stores nothing
    python collective_submit.py --live     # real submission

WHAT LEAVES THIS REPO
Finished numbers only: the game id, the two teams, kickoff, which side the pick is on,
the market line it was priced against, and the probability that pick covers. No source,
no weights, no formulas, no model internals. Everything sent is already public in
data/picks.csv.

KICKOFF
picks.csv carries a date but not a time, and the Collective needs a real kickoff because
only the first submission BEFORE kickoff counts toward the record. Times come from the
nflverse schedule, which is public, needs no key, and is the same source the model itself
uses. If picks.csv ever grows a `kickoff` column this prefers it and skips the fetch.

COVER PROBABILITY
Sent only where the row is a `covers` claim priced against a real book line. A `median`
row is a wins-outright claim with no line behind it, so cover_probability would be a
different quantity wearing the same name. Those rows go without it rather than with a
wrong one.
"""
import argparse, csv, json, os, sys, urllib.request, urllib.error
import datetime as dt
from zoneinfo import ZoneInfo

BASE = "https://iattxbkbufslbauoumga.supabase.co/functions/v1/collective_ingest/v1/projections"
SCHEDULE = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PICKS = os.path.join("data", "picks.csv")
# Slugs, not display names. The onboarding page shows "Moose Metrics (NFL)", which is the
# label; the API binds each key to a slug and rejected that with HTTP 422:
#   This key submits model "moose-metrics", not "Moose Metrics (NFL)"
# The creator slug matches data-collective-host in the embed snippet.
#
# Overridable from the workflow so a rename never needs a code change.
CREATOR = os.environ.get("COLLECTIVE_CREATOR", "").strip() or "mustbemoose"
MODEL = os.environ.get("COLLECTIVE_MODEL", "").strip() or "moose-metrics"
# The key is bound to a sport and the envelope has to name it:
#   HTTP 422  This key submits NFL, the envelope says nothing
# The field is undocumented, so both plausible names are sent. That is safe here rather
# than a guess: the first attempt sent "projections" as the row key and the API objected
# only that "rows" was ABSENT, never that "projections" was unknown, so unrecognised keys
# are ignored. Once a dry run resolves, whichever one is unused can be dropped.
SPORT = os.environ.get("COLLECTIVE_SPORT", "").strip() or "NFL"
# data_origin is a NOT NULL column in their projections table with no default, and their
# handler was not filling it, so a live insert died with Postgres 23502 while the dry run
# looked clean because it never touches the database. The field appears in every dry-run
# response as data_origin: null, which is what gave it away.
#
# Sending it is a workaround for a server-side bug, not a documented field, so the accepted
# value is a guess. Their grading rules separate live first submissions from backfilled
# history, which is the likeliest thing this column records. If the value is rejected the
# error should name the valid ones.
DATA_ORIGIN = os.environ.get("COLLECTIVE_DATA_ORIGIN", "").strip() or "live"
ET = ZoneInfo("America/New_York")

# Full names, not abbreviations. A first dry run sent nflverse codes and every row came
# back quarantined as unknown_game with a null game_id, meaning the Collective could not
# match them to its own schedule. nflverse uses LA for the Rams where most sources use
# LAR, WAS where some use WSH, and JAX where some use JAC, so an abbreviation is the least
# portable thing to key on. A full name is unambiguous under any matching scheme.
FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
    "WSH": "Washington Commanders",
}


def team(code):
    """Full club name. Unknown codes pass through unchanged and get flagged, so a rename
    shows up in the log rather than as a silent quarantine."""
    c = (code or "").strip().upper()
    return FULL.get(c, code.strip())


def kickoffs():
    """game_id -> ISO kickoff. nflverse gametime is Eastern, so it is localised there and
    converted, which keeps September (EDT) and December (EST) both correct."""
    out = {}
    with urllib.request.urlopen(SCHEDULE, timeout=60) as r:
        rows = csv.DictReader(r.read().decode("utf-8", "replace").splitlines())
        for g in rows:
            day, tm = g.get("gameday"), g.get("gametime")
            if not day or not tm:
                continue
            try:
                naive = dt.datetime.strptime(f"{day} {tm}", "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            out[g["game_id"]] = naive.replace(tzinfo=ET).astimezone(dt.timezone.utc) \
                                     .isoformat().replace("+00:00", "Z")
    return out


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build():
    if not os.path.exists(PICKS):
        sys.exit(f"{PICKS} not found. Run the dashboard refresh first.")
    with open(PICKS, newline="") as f:
        rows = list(csv.DictReader(f))
    need_times = not any(r.get("kickoff") for r in rows)
    times = kickoffs() if need_times else {}
    if need_times:
        print(f"kickoff times: {len(times)} games from the nflverse schedule")

    now = dt.datetime.now(dt.timezone.utc)
    out, skipped = [], []
    for r in rows:
        gid = (r.get("game_ref") or r.get("game_id") or "").strip()
        # Already played. The Collective grades first submissions before kickoff, so a
        # settled game is noise at best and a "late" row at worst.
        if (r.get("pick_result") or "").strip():
            continue
        ko = (r.get("kickoff") or "").strip() or times.get(gid)
        if not (gid and r.get("home") and r.get("away") and ko):
            skipped.append((gid or "?", "no kickoff" if not ko else "missing id or teams"))
            continue
        # Kicked off already. Only submissions received BEFORE kickoff are graded, so
        # sending these produces "late" rows and nothing else. A game that has started but
        # whose result has not reached the log yet still has an empty pick_result, so the
        # settled-game filter above does not catch it; this does.
        try:
            if dt.datetime.fromisoformat(ko.replace("Z", "+00:00")) <= now:
                skipped.append((gid, "already kicked off"))
                continue
        except ValueError:
            skipped.append((gid, f"unparseable kickoff {ko!r}"))
            continue
        # season is required and must be an integer. picks.csv carries it; the game_id
        # prefix is the fallback, since nflverse ids start with the season year.
        yr = None
        try:
            yr = int(float(r.get("season") or ""))
        except (TypeError, ValueError):
            head = gid.split("_")[0]
            yr = int(head) if head.isdigit() and len(head) == 4 else None
        if yr is None:
            skipped.append((gid, "no season"))
            continue
        hm, aw = team(r["home"]), team(r["away"])
        for c in (r["home"], r["away"]):
            if c.strip().upper() not in FULL:
                print(f"  WARNING unmapped team code {c!r}, sending it as-is")
        p = {"game_ref": gid, "season": yr, "home_team": hm, "away_team": aw,
             "kickoff": ko}
        side = (r.get("pick_side") or "").strip().lower()
        if side in ("home", "away"):
            p["pick_side"] = side
        # projected_spread is the PUBLISHED pick exactly as Discord shows it, on the
        # PICKED team, with no sign conversion. The Board renders it next to pick_side as
        # "pick away spr +3.5", so the two have to describe the same team: NE +3.5 goes
        # over as pick_side away, projected_spread +3.5.
        #
        # This departs from the onboarding doc, which defines the field on the home team.
        # Converting to home form made the Board read "pick away spr -3.5", which is
        # Seattle's number under New England's label. line_at_submission stays in home
        # form below, so anything grading against the market still gets the documented
        # convention; only the displayed pick uses the picked team's side.
        pl = num(r.get("pick_line"))
        if pl is not None:
            p["projected_spread"] = round(pl, 4)
        for field, col in (("projected_total", "model_total"),
                           ("home_win_probability", "home_win_prob")):
            v = num(r.get(col))
            if v is not None:
                p[field] = round(v, 4)
        line = num(r.get("book_line"))
        prob = num(r.get("pick_prob"))
        # line_at_submission is the HOME team's handicap, negative meaning home favoured,
        # which is already the convention picks.csv stores. No sign flip needed.
        if line is not None:
            p["line_at_submission"] = line
            if prob is not None and (r.get("pick_claim") or "").strip() == "covers":
                p["cover_probability"] = round(prob, 4)
        out.append(p)
    return out, skipped


def post(url, payload, key):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "x-collective-key": key, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="submit for real. Without this it is a dry run and nothing is stored.")
    ap.add_argument("--probe", action="store_true",
                    help="post a deliberately INVALID row to the live endpoint. Safe: their "
                         "validator has to reject it, so nothing can be written. Tells you "
                         "whether a 500 comes from parsing or from persistence.")
    a = ap.parse_args()
    key = os.environ.get("COLLECTIVE_KEY", "").strip()
    if not key:
        sys.exit("COLLECTIVE_KEY is empty. Add it as a repository secret and list it in "
                 "the workflow step's env: block.")

    if a.probe:
        # season must be an integer; sending a string is guaranteed invalid. A 422 back
        # means the validator runs fine on the live route and the 500 is downstream of it,
        # in whatever writes the rows. A 500 back means the route is broken before
        # validation. Either answer localises the fault, and no valid row is ever sent.
        bad = {"creator": CREATOR, "model": MODEL, "sport": SPORT, "league": SPORT,
               "season": "not-an-integer",
               "rows": [{"game_ref": "PROBE-DO-NOT-STORE", "season": "not-an-integer",
                         "home_team": "Seattle Seahawks",
                         "away_team": "New England Patriots",
                         "kickoff": "2026-09-10T00:20:00Z"}]}
        code, text = post(BASE, bad, key)
        print(f"PROBE of the LIVE endpoint with a deliberately invalid payload\n"
              f"=== HTTP {code} ===\n{text[:2000]}\n")
        if code == 422:
            print("422: validation works on the live route, so the 500 is downstream of\n"
                  "it. The crash is in whatever persists the rows, not in parsing.")
        elif code >= 500:
            print("500 even on an invalid payload: the live route is failing before it\n"
                  "validates anything. Nothing you send will get through until it is fixed.")
        return

    games, skipped = build()
    if not games:
        print("no unplayed games with a kickoff, nothing to send")
        return
    # The envelope key is "rows". Confirmed by the API, which answered an earlier guess of
    # "projections" with HTTP 422 invalid_payload: "Envelope must include a non-empty rows
    # array." creator and model are kept as metadata; the key already identifies both, so
    # if a later response objects to them they can go without losing anything.
    # Envelope-level too. The error did not say which level it wanted, and the same
    # ignored-unknown-keys behaviour that made sport/league safe applies here. An NFL
    # season label is constant through its own playoffs, so one value per envelope is
    # never ambiguous; if a slate ever straddled two, the most common wins and it says so.
    yrs = [g["season"] for g in games]
    season = max(set(yrs), key=yrs.count)
    if len(set(yrs)) > 1:
        print(f"note: rows span seasons {sorted(set(yrs))}, envelope says {season}")
    payload = {"creator": CREATOR, "model": MODEL, "sport": SPORT, "league": SPORT,
               "season": season, "data_origin": DATA_ORIGIN, "rows": games}

    live = a.live or os.environ.get("COLLECTIVE_LIVE", "").strip().lower() in ("1", "true", "yes")
    print(f"\n=== {len(games)} game(s) {'LIVE' if live else 'DRY RUN'} ===")
    # A compact table rather than a wall of JSON, so every game is visible in the log
    # instead of the first few and a truncation mark. One full row follows as a shape check.
    print(f"{'game_ref':<20} {'matchup':<11} {'kickoff':<21} {'side':>5} "
          f"{'mdlspr':>6} {'homeW':>6} {'line':>6} {'cover':>7}")
    for g in games:
        ln = g.get("line_at_submission")
        cv = g.get("cover_probability")
        print(f"{g['game_ref']:<20} {g['game_ref'].split('_', 2)[2]:<11} "
              f"{g['kickoff']:<21} {g.get('pick_side', '-'):>5} "
              f"{('-' if 'projected_spread' not in g else format(g['projected_spread'], '+g')):>6} "
              f"{('-' if 'home_win_probability' not in g else format(g['home_win_probability'], '.3f')):>6} "
              f"{('-' if ln is None else format(ln, '+g')):>6} "
              f"{('-' if cv is None else format(cv, '.4f')):>7}")
    for gid, why in skipped:
        print(f"  skipped {gid}: {why}")
    print("\nfirst row as sent:\n" + json.dumps(games[0], indent=2))

    url = BASE if live else BASE + "/dry-run"
    code, text = post(url, payload, key)
    print(f"\n=== HTTP {code} ===\n{text[:4000]}")
    if code >= 500 or code == 0:
        # A 5xx says the request reached them and something broke while handling it. It
        # does NOT say whether the write happened, and only the first submission per game
        # before kickoff is graded, so a blind retry can spend that status on rows that
        # already have one. Verify before resending; a dry run is free and its duplicate
        # flag answers the question.
        print("\nSERVER ERROR, on their side: the same envelope validates on /dry-run.")
        # A Postgres constraint code means the insert was refused and the transaction rolled
        # back, so nothing persisted and a retry is safe. Any other 5xx leaves the write
        # state unknown, and a blind retry can spend the first-submission status that
        # grading depends on.
        if '"db_code"' in text:
            print("A db_code means the database REFUSED the insert, so nothing was stored\n"
                  "and retrying is safe. Send the db_message to whoever runs the Collective;\n"
                  "a not-null or constraint failure on their own table is their fix.")
        else:
            print("No database detail returned, so it is unknown whether rows were stored.\n"
                  "Do NOT resubmit until you know. Run with live=false and check the\n"
                  "duplicate flag, and check the Collective dashboard for a submission at\n"
                  "the timestamp above.")
    if code == 422 and "This key submits" in text:
        # Matched on "not" before, and "nothing" contains "not", so a missing-sport error
        # printed advice about creator and model slugs. Match the actual phrase instead.
        print(f"\nIdentity sent: creator={CREATOR!r} model={MODEL!r} sport={SPORT!r}.\n"
              "Override with COLLECTIVE_CREATOR, COLLECTIVE_MODEL or COLLECTIVE_SPORT in\n"
              "the workflow env to whatever the message above names.")
    if code >= 300 or code == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
