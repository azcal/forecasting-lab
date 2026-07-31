import json, re, gzip, os
import pandas as pd
OUT = "/home/claude/mma"
RAW = os.path.join(OUT, "raw.jsonl.gz")


def find_templates(txt, name="MMAevent bout"):
    key = "{{" + name
    i = 0
    while True:
        s = txt.find(key, i)
        if s < 0:
            return
        j = s + 2
        depth = 1
        while j < len(txt) and depth:
            if txt.startswith("{{", j):
                depth += 1; j += 2
            elif txt.startswith("}}", j):
                depth -= 1; j += 2
            else:
                j += 1
        yield txt[s + 2:j - 2]
        i = j


def split_fields(body):
    parts, buf, sq, br, i = [], "", 0, 0, 0
    while i < len(body):
        if body.startswith("[[", i):
            sq += 1; buf += "[["; i += 2
        elif body.startswith("]]", i):
            sq -= 1; buf += "]]"; i += 2
        elif body.startswith("{{", i):
            br += 1; buf += "{{"; i += 2
        elif body.startswith("}}", i):
            br -= 1; buf += "}}"; i += 2
        elif body[i] == "|" and sq <= 0 and br <= 0:
            parts.append(buf); buf = ""; i += 1
        else:
            buf += body[i]; i += 1
    parts.append(buf)
    return parts


def clean(s):
    s = re.sub(r"<ref[^>]*?/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<[^>]*>", "", s)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[|\]\]", "", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = s.replace("\u2019", "'")
    s = re.sub(r"\(c\)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_bouts(txt):
    out = []
    for body in find_templates(txt):
        f = [x.strip() for x in split_fields(body)]
        # field 0 is always the template name fragment, "MMAevent bout"
        if f and f[0].lower().replace(" ", "").startswith("mmaeventbout"):
            f = f[1:]
        if len(f) < 7:
            continue
        wc, f1, sep, f2, method, rnd, tm = f[:7]
        if sep.strip().lower().rstrip(".") not in ("def", "vs"):
            continue
        out.append({"weight": clean(wc), "f1": clean(f1),
                    "sep": sep.strip().lower().rstrip("."), "f2": clean(f2),
                    "method": clean(method), "round": clean(rnd), "time": clean(tm),
                    "notes": clean(f[7]) if len(f) > 7 else ""})
    return out


if __name__ == "__main__":
    # Only runs when invoked directly. The daily runner imports the
    # parsing helpers above and must not trigger a full re-parse.
    rows, empty = [], []
    with gzip.open(RAW, "rt") as fh:
        for line in fh:
            e = json.loads(line)
            b = parse_bouts(e["wikitext"])
            if not b:
                empty.append((e["promotion"], e["date"][:4], e["event"], len(e["wikitext"])))
            for x in b:
                rows.append({**x, "event": e["event"], "promotion": e["promotion"], "date": e["date"]})

    F = pd.DataFrame(rows)
    F["date"] = pd.to_datetime(F.date)
    F.to_csv(os.path.join(OUT, "fights_raw.csv"), index=False)
    print(f"bouts parsed: {len(F)}")
    print(F.promotion.value_counts().to_string())
    print()
    Z = pd.DataFrame(empty, columns=["promo", "year", "event", "bytes"])
    print(f"events yielding nothing: {len(Z)}  (page missing: {(Z.bytes==0).sum()}, "
          f"page present but no template: {(Z.bytes>0).sum()})")
    print(Z[Z.bytes > 0].groupby("promo").size().to_string())
    print()
    u = F[F.promotion == "UFC"]
    print("UFC bouts by year:")
    print(u.groupby(u.date.dt.year).size().tail(18).to_string())
