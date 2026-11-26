#!/usr/bin/env python3
"""Decide whether this workflow tick should do real work.

Why this exists (2026-11-26 audit). GitHub starts scheduled workflows late, and through
November the delay ran from 20 minutes to 10 hours with a median near 5. A cron line of
"9:00am" produced boards at 2:30pm and "4:00pm" produced boards at 10pm, after the games
had started. Moving the cron earlier does not help because the delay is not constant.

So the workflow now ticks every hour and this script answers one question: is there a
slot whose target time has passed, whose window is still open, and that has not been run
yet? If yes the work job runs and records the slot in a marker file when it commits. If no
the tick ends in a few seconds. A late tick still lands within about an hour of the
target, and a run that fails leaves no marker, so the next tick retries it.

Slots are local wall-clock times, so daylight saving needs no second cron line.

Environment
  SLOTS         comma list. Each item is [name=][day@]HH:MM, day is mon..sun.
                "am=09:00,pm=16:00"   "week=sun@06:00,thu=thu@09:00"
  SLOT_TZ       IANA zone, default America/Edmonton
  SLOT_WINDOW_H hours a slot stays runnable after its target, default 4
  SLOT_MARKER   file holding the last completed slot id, default logs/last_slot.txt
  SLOT_SEASON   optional "MM-DD..MM-DD" (may wrap the new year). Outside it nothing runs.
  SLOT_EVENT    github.event_name. Anything other than "schedule" always runs as "manual".

Writes run, slot and name to $GITHUB_OUTPUT. Never fails the job on bad state: an
unreadable marker means "not run yet", which costs one extra board, not a missed one.
"""
import datetime as dt
import os
import sys
from zoneinfo import ZoneInfo

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def parse_slots(spec):
    out = []
    for i, item in enumerate(x.strip() for x in spec.split(",") if x.strip()):
        name, rest = item.split("=", 1) if "=" in item else (f"s{i + 1}", item)
        day, hhmm = rest.split("@", 1) if "@" in rest else (None, rest)
        if day is not None:
            day = DAYS.index(day.strip().lower()[:3])
        h, m = hhmm.strip().split(":")
        out.append((name.strip(), day, dt.time(int(h), int(m))))
    return out


def in_season(now, season):
    if not season:
        return True
    a, b = [x.strip() for x in season.split("..")]
    md = now.strftime("%m-%d")
    return (a <= md <= b) if a <= b else (md >= a or md <= b)


def decide(now, slots, window_h, marker_text, season=""):
    """(run, slot_id, name). slot_id is the local target time, which is what makes a slot
    unique: 'YYYY-MM-DD HH:MM name'."""
    if not in_season(now, season):
        return False, "", ""
    best = None
    for name, day, t in slots:
        for back in (0, 1):                      # yesterday too: a late-evening slot's
            d = (now - dt.timedelta(days=back)).date()   # window can cross midnight
            if day is not None and d.weekday() != day:
                continue
            target = dt.datetime.combine(d, t, tzinfo=now.tzinfo)
            if target <= now < target + dt.timedelta(hours=window_h):
                if best is None or target > best[0]:
                    best = (target, name)
    if best is None:
        return False, "", ""
    slot_id = f"{best[0]:%Y-%m-%d %H:%M} {best[1]}"
    done = {ln.strip() for ln in marker_text.splitlines() if ln.strip()}
    return slot_id not in done, slot_id, best[1]


def main():
    event = os.environ.get("SLOT_EVENT", "schedule")
    tz = ZoneInfo(os.environ.get("SLOT_TZ", "America/Edmonton"))
    now = dt.datetime.now(tz)
    if event != "schedule":
        run, slot, name = True, "manual", "manual"
    else:
        marker = os.environ.get("SLOT_MARKER", "logs/last_slot.txt")
        try:
            text = open(marker).read()
        except OSError:
            text = ""
        run, slot, name = decide(now, parse_slots(os.environ.get("SLOTS", "")),
                                 float(os.environ.get("SLOT_WINDOW_H", "4")), text,
                                 os.environ.get("SLOT_SEASON", ""))
    print(f"slot gate {now:%Y-%m-%d %H:%M %Z}: event={event} run={run} slot={slot or '-'}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if run else 'false'}\nslot={slot}\nname={name}\n")


if __name__ == "__main__":
    sys.exit(main())
