"""
One board line, one format, every sport, Telegram and dashboard.

    2026-09-14  KC @ LAC        LAC +2.5        find 1.91 / -110 or better

Columns: date, game, what the model favours, and the worst price worth taking. The price
is the same min_odds rule every board already uses, shown in decimal and American so it can
be compared to whatever a book displays without conversion.
"""
import math

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
    return (f"{date}  {game:<{w_game}} {pick:<{w_pick}} "
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
    """Column widths that fit the slate, so a board never ragged-wraps in Telegram."""
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
