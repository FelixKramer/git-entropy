#!/usr/bin/env python3
"""
git-entropy — how much information is actually in your commit history?

Not how many commits. Not how many words. How much *information* — the number of bytes you
would genuinely need to transmit your project's entire commit history to someone else.

The answer is usually humbling.

Method, and why the obvious version is wrong
--------------------------------------------
The naive approach is to compress the commit log and report the size. That number is real, but
most of it is *vocabulary*: the fact your team writes "fix" four hundred times is a property of
the word distribution, not information carried by any individual commit.

So this measures three things instead:

1. TOTAL INFORMATION — the compressed size of the whole history. Honest and concrete: this is
   what it would cost to send someone every commit message you have ever written.

2. NEW INFORMATION PER COMMIT — the incremental cost of appending each message given every
   message before it. A commit that repeats what the history already contains costs almost
   nothing to transmit, because it told you almost nothing. Computed in one pass with a
   streaming compressor, so it is exact rather than estimated.

3. STRUCTURE vs VOCABULARY — the part people skip. We rebuild the corpus with the word
   frequencies preserved exactly but the arrangement destroyed, then compress that. Whatever
   survives is genuine structure — recurring phrases, templates, conventions. Whatever does not
   was only ever your team's habitual word choice wearing a disguise.

Nothing here is novel mathematics. Permutation nulls and compression-based information estimates
are old and well understood. What is uncommon is bothering to run the null at all, which is the
difference between measuring information and measuring your own vocabulary back at yourself.

Usage:
    python git_entropy.py [PATH] [--limit N] [--author NAME] [--json]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import zlib
from collections import Counter

SEP = "\x00\x00"          # message separator that cannot occur in a commit message
LEVEL = 9

# Windows terminals still default to cp1252, which cannot encode box-drawing or block glyphs.
# A stranger's first run must never be a traceback, so ask for UTF-8 and fall back to ASCII if
# the terminal refuses.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _unicode_ok():
    try:
        "─█".encode(sys.stdout.encoding or "ascii")
        return True
    except Exception:
        return False


UNI = _unicode_ok()
G = {
    "h":  "─" if UNI else "-",
    "tl": "┌" if UNI else "+",
    "tr": "┐" if UNI else "+",
    "bl": "└" if UNI else "+",
    "br": "┘" if UNI else "+",
    "v":  "│" if UNI else "|",
    "on": "█" if UNI else "#",
    "off": "·" if UNI else ".",
    "x":  "×" if UNI else "x",
    "dot": "·" if UNI else "-",
}


# ----------------------------------------------------------------------------- git


def read_commits(path=".", limit=0, author=None, keep_trailers=False):
    """Commit subjects + bodies, oldest first, so 'new information' means what it says."""
    fmt = "%H%x1f%an%x1f%ad%x1f%B%x1e"
    cmd = ["git", "-C", path, "log", "--reverse", f"--pretty=format:{fmt}", "--date=short"]
    if limit:
        cmd += [f"-n{limit}"]
    if author:
        cmd += [f"--author={author}"]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=True).stdout
    except FileNotFoundError:
        sys.exit("error: git is not on PATH")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: not a git repository, or git failed: {(e.stderr or '').strip()[:200]}")

    out = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split("\x1f")
        if len(parts) < 4:
            continue
        sha, an, ad, body = parts[0], parts[1], parts[2], parts[3]
        msg = (body if keep_trailers else strip_trailers(body)).strip()
        if msg:
            out.append({"sha": sha[:8], "author": an, "date": ad, "msg": msg})
    return out


TRAILER = re.compile(
    r"^\s*(?:co-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|reported-by|"
    r"suggested-by|cc|closes|fixes|refs|change-id|git-svn-id)\s*:",
    re.IGNORECASE)


def strip_trailers(body):
    """Drop git trailers before measuring.

    They are machine-generated boilerplate attached to the message, not information anyone
    wrote, and leaving them in measures your tooling's signature rather than your team's.
    A repo that appends the same trailer to every commit would otherwise look far more
    repetitive than it is. Use --keep-trailers if you actually want them counted.
    """
    return "\n".join(l for l in body.splitlines() if not TRAILER.match(l))


# ----------------------------------------------------------------------------- measurement


def total_information(messages):
    """Compressed size of the whole history — what it would cost to transmit all of it."""
    blob = SEP.join(messages).encode("utf-8", "replace")
    return len(zlib.compress(blob, LEVEL)), len(blob)


def incremental_information(messages):
    """Exact per-commit cost of appending each message GIVEN everything before it.

    One streaming pass: flush the compressor after each message and measure how much output it
    produced. A message the history already predicts costs almost nothing — which is precisely
    the sense in which it told you nothing.
    """
    co = zlib.compressobj(LEVEL)
    costs = []
    for m in messages:
        chunk = co.compress((m + SEP).encode("utf-8", "replace"))
        chunk += co.flush(zlib.Z_SYNC_FLUSH)
        costs.append(len(chunk))
    return costs


def shuffled_null(messages, seed=0):
    """Same words, same frequencies, arrangement destroyed.

    Preserving the exact word multiset is the whole point: any size difference that remains
    cannot be explained by vocabulary, so it is structure — templates, recurring phrases,
    conventions. This is the control most people skip, and skipping it is how you end up
    reporting your own word-frequency table as if it were insight.
    """
    rng = random.Random(seed)
    words = SEP.join(messages).split()
    rng.shuffle(words)
    # rebuild messages of the same word-lengths so per-message overhead matches too
    out, i = [], 0
    for m in messages:
        n = len(m.split())
        out.append(" ".join(words[i:i + n]))
        i += n
    blob = SEP.join(out).encode("utf-8", "replace")
    return len(zlib.compress(blob, LEVEL))


WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def phrase_counts(messages, n=2, top=6):
    c = Counter()
    for m in messages:
        w = [x.lower() for x in WORD.findall(m)]
        for i in range(len(w) - n + 1):
            c[" ".join(w[i:i + n])] += 1
    return c.most_common(top)


# ----------------------------------------------------------------------------- output


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n/1:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def bytes_h(n):
    if n < 1024:
        return f"{n:,} B"
    if n < 1024 ** 2:
        return f"{n/1024:,.1f} KB"
    return f"{n/1024**2:,.2f} MB"


def bar(frac, width=28):
    filled = max(0, min(width, round(frac * width)))
    return G["on"] * filled + G["off"] * (width - filled)


def report(commits, args):
    msgs = [c["msg"] for c in commits]
    words = sum(len(m.split()) for m in msgs)
    comp, raw = total_information(msgs)
    costs = incremental_information(msgs)
    null = shuffled_null(msgs)

    order = sorted(range(len(costs)), key=lambda i: costs[i])
    med = sorted(costs)[len(costs) // 2]
    free = sum(1 for c in costs if c <= 8)          # essentially told us nothing
    structure = max(0.0, (null - comp) / null) if null else 0.0

    W = 64
    print()
    print("  git-entropy".ljust(W) + "\n  " + G["h"] * (W - 2))
    print(f"  {len(commits):,} commits {G['dot']} {words:,} words "
          f"{G['dot']} {bytes_h(raw)} of text\n")

    inner = W - 4

    def boxline(text):
        return "  " + G["v"] + "  " + text.ljust(inner - 2) + G["v"]

    print("  " + G["tl"] + G["h"] * inner + G["tr"])
    print(boxline("ACTUAL INFORMATION CONTENT"))
    print(boxline(bytes_h(comp)))
    print(boxline("the whole history would fit in this"))
    print("  " + G["bl"] + G["h"] * inner + G["br"] + "\n")

    print(f"  compression         {bar(comp / raw)}  {raw/comp:,.0f}{G['x']} smaller than the text")
    print(f"  new info per commit {bar(min(1, med / 60))}  {med} bytes (median)")
    print(f"  told us nothing new {bar(free / len(costs))}  {free:,} commits ({free/len(costs)*100:.0f}%)")
    print(f"  real structure      {bar(structure)}  {structure*100:.0f}% survives word-shuffling\n")

    print("  LEAST INFORMATIVE COMMITS")
    for i in order[:5]:
        subj = commits[i]["msg"].splitlines()[0][:44]
        print(f"    {costs[i]:>4} B   {commits[i]['sha']}  {subj}")

    print("\n  MOST INFORMATIVE COMMITS")
    for i in order[-3:][::-1]:
        subj = commits[i]["msg"].splitlines()[0][:44]
        print(f"    {costs[i]:>4} B   {commits[i]['sha']}  {subj}")

    ph = phrase_counts(msgs)
    if ph:
        print("\n  MOST REPEATED PHRASES")
        for p, n in ph:
            print(f"    {n:>4}{G['x']}   {p}")

    print("\n  " + G["h"] * (W - 2))
    print("  What this measures: the bytes needed to transmit your history, and how much each")
    print("  commit added given the ones before it. What it does not measure: whether any of")
    print("  it was worth writing. A 6-byte commit can be the right commit.")
    print(f"  The {structure*100:.0f}% figure is what survives shuffling the words while keeping")
    print("  their frequencies identical - so it is structure, not vocabulary.\n")

    if args.json:
        print(json.dumps({
            "commits": len(commits), "words": words, "raw_bytes": raw,
            "information_bytes": comp, "ratio": raw / comp,
            "median_new_bytes": med, "uninformative_commits": free,
            "structure_fraction": structure,
        }, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="git-entropy",
        description="How much information is actually in your commit history?")
    ap.add_argument("path", nargs="?", default=".", help="path to a git repository (default: .)")
    ap.add_argument("--limit", type=int, default=0, help="only the most recent N commits")
    ap.add_argument("--author", help="filter to one author")
    ap.add_argument("--json", action="store_true", help="also emit machine-readable JSON")
    ap.add_argument("--keep-trailers", action="store_true",
                    help="count git trailers (Co-authored-by, Signed-off-by, ...) as information")
    args = ap.parse_args(argv)

    commits = read_commits(args.path, args.limit, args.author, args.keep_trailers)
    if len(commits) < 5:
        sys.exit("error: need at least 5 commits with messages to say anything meaningful")
    report(commits, args)


if __name__ == "__main__":
    main()
