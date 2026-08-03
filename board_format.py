"""
One board line, one format, every sport, Discord and dashboard.

    2026-09-14  KC @ LAC        LAC +2.5        find 1.91 / -110 or better

Columns: date, game, what the model favours, and the worst price worth taking. The price
is the same min_odds rule every board already uses, shown in decimal and American so it can
be compared to whatever a book displays without conversion.
"""
import math

# Bumped whenever this file changes. Every runner asserts on it at import, so a repo
# holding an old copy fails in the workflow's import guard instead of rendering wrong.
VERSION = "2026-08-03"

MIN_EDGE = 0.05

# Worst decimal price a board will print. Past this the model's number implies odds no book
# posts, so the row is noise rather than a bet. Books do reach -400 and -500 on heavy soccer
# and esports favourites, which is why the floor sits there rather than somewhere tighter.
# Measured cut at 1.25: Soccer 17%, CS2 8%, MMA 0.3%.
PRICE_FLOOR = 1.25


def min_odds(p, edge=MIN_EDGE):
    """Worst decimal price worth taking at probability p."""
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + edge) / p * 100) / 100.0


def american(dec):
    """Decimal to American. 2.00 is the pivot: above it pays plus, below it lays minus."""
    dec = float(dec)
    if dec >= 2.0:
        return f"+{round((dec - 1) * 100):d}"
    return f"{round(-100 / (dec - 1)):d}"


def price(p, edge=MIN_EDGE):
    """'1.91 / -110' for a probability."""
    d = min_odds(p, edge)
    return f"{d:.2f} / {american(d)}"


def line(date, game, pick, p, edge=MIN_EDGE, w_game=26, w_pick=16):
    """A single board row. `pick` is the outcome in the reader's language: a team for a
    moneyline, a team plus handicap for a spread, a fighter for MMA."""
    # Two spaces between fields, not one. The column padding is sized to the widest
    # entry on the slate, so that row gets zero padding; with single-space separators it
    # ends up one space from the next field and notify.BOARD, which needs two or more,
    # fails to match. The row then falls through and renders as an unstyled run-on line.
    return (f"{date}  {game:<{w_game}}  {pick:<{w_pick}}  "
            f"find {price(p, edge)} or better")


STD_SPREAD_ODDS = 1.91          # -110, how spreads are normally priced


def spread_threshold(cover_fn, i, grid, edge=MIN_EDGE, odds=STD_SPREAD_ODDS):
    """The worst line worth taking on each side at standard spread pricing.

    At the model's own line P(cover) is 0.50 by construction, so quoting a price there
    would print 2.10 / +110 on every game and say nothing. What is actionable is the line
    at which the normal -110 becomes worth taking, which needs p >= (1+edge)/odds.

    Returns (home_line, away_line) in margin space, or None where no line qualifies.
    """
    need = (1 + edge) / odds
    ph = [cover_fn(k)[i] for k in grid]
    hi = [k for k, p in zip(grid, ph) if p >= need]
    lo = [k for k, p in zip(grid, ph) if (1 - p) >= need]
    return (max(hi) if hi else None, min(lo) if lo else None)


def spread_pick(home, away, line_margin):
    """Margin-space line to a quoted pick. A book quotes the handicap added to a team's
    score, which is the negative of the margin threshold for the home side. Getting this
    backwards hands you the opposite bet."""
    if line_margin >= 0:
        return f"{home} -{abs(line_margin):.1f}"
    return f"{away} -{abs(line_margin):.1f}"


def widths(games, picks):
    """Column widths that fit the slate, so a board never ragged-wraps."""
    return (max([len(g) for g in games] + [10]),
            max([len(p) for p in picks] + [8]))


def build_board(rows, floor=PRICE_FLOOR, edge=MIN_EDGE):
    """Format a full board and report what the floor removed.

    rows: iterable of (date, game, pick, probability).
    Returns (lines, skipped). A skipped game is not a game the model has no view on, it is
    one whose price is unfindable, so the count is printed rather than the row silently
    disappearing.
    """
    keep = [(d, g, p, q) for d, g, p, q in rows if min_odds(q, edge) >= floor]
    skipped = len(list(rows)) - len(keep) if not hasattr(rows, "__len__") else len(rows) - len(keep)
    if not keep:
        out = []
    else:
        wg = max(len(g) for _, g, _, _ in keep)
        wp = max(len(p) for _, _, p, _ in keep)
        out = [line(d, g, p, q, edge, wg, wp) for d, g, p, q in keep]
    if skipped:
        out.append(f"({skipped} game{'s' if skipped != 1 else ''} skipped, "
                   f"price floor {floor:.2f} / {american(floor)})")
    return out, skipped


# ---------------------------------------------------------------- combinations
def mma_board(p_a_ko, p_a_sub, p_a_dec, p_b_ko, p_b_sub, p_b_dec, a, b,
              floor=PRICE_FLOOR, edge=MIN_EDGE):
    """Two lines for a fight: the moneyline lean and the distance lean.

    Decision is the distance. KO/TKO and submission are both finishes, so whichever side
    of that is likelier becomes the second line, quoted as the yes/no market books post
    alongside the winner.

    The two legs were previously also offered combined, as the same-game parlay a book
    posts on one ticket, with a bracket showing what independent pricing implied. That
    row is gone. It was the only place a bracket appeared, so `ROW`'s optional note group
    in notify.py now has no producer; it is left in place for other boards to use.
    """
    pA = p_a_ko + p_a_sub + p_a_dec
    pB = p_b_ko + p_b_sub + p_b_dec
    dist = p_a_dec + p_b_dec
    fin = p_a_ko + p_a_sub + p_b_ko + p_b_sub

    win_name, win_p = (a, pA) if pA >= pB else (b, pB)
    way_name, way_p = ("goes the distance", dist) if dist >= fin else ("ends inside", fin)

    rows = [(win_name, win_p), (way_name, way_p)]
    out, w = [], max(len(r[0]) for r in rows)
    for label, p in rows:
        if min_odds(p, edge) < floor:
            continue
        out.append(f"  {label:<{w}} {p:>5.1%}  find {price(p, edge)} or better")
    return out


# ---------------------------------------------------------------- live entry
# Windows measured from each board's own walk-forward pregame probability, not a stand-in
# rating. That matters: the production models pick more resilient short favourites than a
# bare Elo does, because they can see availability and rest. Measured on the same games a
# bare Elo flagged, down 3-7 after Q1 read 58.9%; measured on the games these models flag,
# it reads 63.1%.
#
# Ceiling is a fixed 10-point tolerance below the historical rate, not a confidence
# interval. It answers "has this price drifted far enough to be telling me something the
# scoreboard is not", which does not depend on how many games happened to be in the cell.
#
#   sport -> checkpoint, [(deficit_low, deficit_high, take_from, take_to), ...]
LIVE_MAX = 1.50
TRAP_DROP = 0.10
LIVE_WINDOWS = {
    "NBA":   ("Q1", [(-11, -7, 1.72, 1.96), (-7, -3, 1.67, 1.88), (-3, 0, 1.46, 1.61)]),
    "NFL":   ("Q1", [(-10, -7, 2.00, 2.35), (-7, -3, 1.66, 1.87), (-3, 0, 1.36, 1.49)]),
    "NCAAF": ("Q1", [(-14, -7, 1.96, 2.29), (-7, -3, 1.49, 1.65), (-3, 0, 1.36, 1.48)]),
    "WNBA":  ("Q1", [(-9, -5, 1.69, 1.91), (-5, 0, 1.65, 1.86)]),
}

# Measured and rejected. These are findings about the sports, not gaps in the build.
#
#   NHL     down two after one period: 46.1%. Down three: 31.6%. Only the one-goal row
#           cleared 50% and its window was two cents wide.
#   Soccer  down one at half: 35.9%. Down two: 12.0%. Only level-at-half survived.
#   CS2     favourite losing map one wins the series 38.9% across 558 best-of-threes.
#           A Bo3 turns map one into a near-elimination.
#   MMA     no scoreboard between rounds. Judging is not observable mid-fight.
#
# The pattern: comebacks are live where scoring is frequent and continuous, and dead where
# it is rare (hockey, soccer) or where the format segments into must-wins (CS2).


def live_note(sport, p_pre, indent="      "):
    """Live-entry addendum for a pregame board line.

    Only for favourites already too short to take at the open, which is the premise: you are
    not betting the opener, you are waiting for a price the scoreboard creates. Returns []
    where the sport has no measured window or the pregame price is long enough to just take.

    NHL, Soccer, CS2 and MMA are absent because they were measured and did not qualify.
    See the note above LIVE_WINDOWS.
    """
    if sport not in LIVE_WINDOWS or min_odds(p_pre) >= LIVE_MAX:
        return []
    cp, rows = LIVE_WINDOWS[sport]
    return [f"{indent}or live {a:.2f}-{b:.2f} ({american(a)} to {american(b)}) "
            f"after {cp} if down {abs(hi) if hi else 0}-{abs(lo)}"
            for lo, hi, a, b in rows]
