#!/usr/bin/env python3
"""Codegen-size report over the cgs JSON dumps, old vs current.

Reads the cgsReport dumps under -d:TTT_CgsDump,
cur_reports/<runner>.json and old_reports/<runner>.json,
and prints the complete delta report:

- global summary: per-runner totals and the biggest kernel movers
- per runner: old/current metadata, kernel cost table with deltas,
  new and gone kernels, stats and bucket ladder deltas, and the top
  attribution movers by Nim origin

Usage: python3 report_cgs.py [--dir DIR] [--runner NAME]
--dir defaults to this script's dumps directory.
"""
import argparse
import json
import sys
from pathlib import Path

BUCKETS = ["<=5", "<=30", "<=70", "<=150", "<=300", "<=700", "<=1400", ">1400"]
STAT_FIELDS = ["bytes", "lines", "types", "vars", "funcs", "inline",
               "nonInline", "calls"]


def fmt(d):
    """Signed delta, + for growth."""
    return f"+{d}" if d >= 0 else str(d)


def pct(old, cur):
    """Percent change of cur over old, one decimal."""
    if old == 0:
        return "n/a"
    return f"{(cur - old) * 100.0 / old:+.1f}%"


def load(path):
    """Parsed JSON dump, None when the file is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def kernels(doc):
    """{name: row} keyed on kernel name."""
    return {k["name"]: k for k in doc["kernels"]}


def origin_locs(doc):
    """{origin: summed loc} over every kernel's attribution."""
    locs = {}
    for k in doc["kernels"]:
        for o in k["attribution"]:
            locs[o["origin"]] = locs.get(o["origin"], 0) + o["loc"]
    return locs


def meta_line(tag, doc):
    """One-line metadata of a dump, branch@short-commit and datetime."""
    if doc is None:
        return f"{tag}: (absent)"
    return (f"{tag}: {doc['branch']}@{doc['commit'][:8]} {doc['datetime']}")


def runner_report(name, cur, old):
    """Prints the delta report of one runner."""
    print(f"\n=== {name} ===")
    print(meta_line("old ", old))
    print(meta_line("curr", cur))
    if cur is None:
        print("no current dump, nothing to report")
        return
    cur_k, old_k = kernels(cur), (kernels(old) if old else {})
    hdr = f"| {'kernel':<32}| {'old':>9}| {'curr':>9}| {'delta':>9}| {'pct':>8}|"
    sep = "|" + "-" * len(hdr[1:-1]) + "|"
    if old is not None:
        print(hdr)
        print(sep)
    movers = []
    for name_k in sorted(set(cur_k) | set(old_k)):
        c, o = cur_k.get(name_k), old_k.get(name_k)
        if c is None:
            print(f"| {name_k:<32}| {'-':>9}| {'gone':>9}|")
            continue
        if o is None:
            tag = "new " if old is not None else ""
            print(f"| {name_k:<32}| {'-':>9}| {c['cost']:>9}|{tag:>10}|")
            continue
        d = c["cost"] - o["cost"]
        print(f"| {name_k:<32}| {o['cost']:>9}| {c['cost']:>9}|"
              f" {fmt(d):>9}| {pct(o['cost'], c['cost']):>8}|")
        movers.append((d, o["cost"], c["cost"], name_k, o["marginal"],
                       c["marginal"]))
    to, tc = cur["totals"], (old["totals"] if old else None)
    if old is not None:
        d = to["cost"] - tc["cost"]
        print(f"| {'total':<32}| {tc['cost']:>9}| {to['cost']:>9}|"
              f" {fmt(d):>9}| {pct(tc['cost'], to['cost']):>8}|")
        movers.sort(key=lambda m: -abs(m[0]))
        if movers and movers[0][0] != 0:
            print(" biggest movers:")
            for d, oc, cc, kn, _, _ in movers[:5]:
                if d == 0:
                    break
                print(f"   {kn}: {oc} -> {cc} ({fmt(d)}, {pct(oc, cc)})")
    else:
        print(f"| total current cost {to['cost']}, first recorded dump")

    # stats and bucket ladders, delta only when old exists
    secs = [("stats", cur["totals"]["stats"],
             old["totals"]["stats"] if old else None, STAT_FIELDS)]
    for bk, key in [("function LOC", "functionLOC"),
                    ("overload groups", "overloadGroups"),
                    ("overload members", "overloadMembers")]:
        secs.append((bk, cur["buckets"][key],
                     old["buckets"][key] if old else None, BUCKETS))
    for title, cur_s, old_s, fields in secs:
        print(f" {title}:")
        for f in fields:
            c = cur_s.get(f, 0)
            if old_s is None:
                print(f"   {f:<8} {c:>9}")
            else:
                o = old_s.get(f, 0)
                print(f"   {f:<8} {o:>9} -> {c:>9} {fmt(c - o):>8}"
                      f" {pct(o, c):>8}")
    fo, fc = (old["buckets"]["families"] if old else None,
              cur["buckets"]["families"])
    if old is None:
        print(f"   families {fc}")
    else:
        print(f"   families {fo:>9} -> {fc:>9} {fmt(fc - fo):>8}")

    # attribution movers by Nim origin, summed loc over kernels
    cur_o, old_o = origin_locs(cur), (origin_locs(old) if old else {})
    moves = [(cur_o.get(o, 0) - old_o.get(o, 0), o)
             for o in set(cur_o) | set(old_o)]
    moves = [m for m in moves if m[0] != 0]
    if moves and old is not None:
        print(" attribution movers (loc by Nim origin):")
        moves.sort(key=lambda m: -abs(m[0]))
        for d, o in moves[:5]:
            print(f"   {o}: {old_o.get(o, 0)} -> {cur_o.get(o, 0)} ({fmt(d)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(Path(__file__).parent / "dumps"))
    ap.add_argument("--runner", default=None, help="one runner, else all")
    args = ap.parse_args()
    d = Path(args.dir)
    if not (d / "cur_reports").is_dir():
        sys.exit(f"dumps cur_reports directory not found: {d / 'cur_reports'}")
    names = sorted({p.name[:-5] for p in (d / "cur_reports").glob("*.json")})
    if args.runner:
        names = [n for n in names if n == args.runner]
        if not names:
            sys.exit(f"no dump for runner {args.runner}")
    if not names:
        sys.exit("no dumps found, run the cgs runners with -d:TTT_CgsDump")
    grand_old = grand_cur = 0
    have_old = False
    for n in names:
        cur, old = load(d / "cur_reports" / f"{n}.json"), load(d / "old_reports" / f"{n}.json")
        if old is not None:
            have_old = True
            grand_old += old["totals"]["cost"]
        if cur is not None:
            grand_cur += cur["totals"]["cost"]
        runner_report(n, cur, old)
    print("\n=== all runners ===")
    if have_old:
        print(f"total cost {grand_old} -> {grand_cur} "
              f"({fmt(grand_cur - grand_old)}, {pct(grand_old, grand_cur)})")
    else:
        print(f"total current cost {grand_cur}, first recorded dump")


if __name__ == "__main__":
    main()
