#!/usr/bin/env python3
"""Fixture file size check over the files ADDED by a commit range.

Usage: check_fixture_filesize.py <repo> <base> <head-or-''> [paths...]

Modes:
  - Diff mode (head given or head is '', like scan_added.py): every file
    ADDED by `git diff base..head` (base..worktree when head is '') that
    qualifies as a fixture file is checked against the size caps below.
    Worktree mode (head '') also covers UNTRACKED new files: a brand-new
    fixture left unstaged is invisible to the diff and still counts.
  - Sweep mode (base is '-', head a ref or the empty string): every
    TRACKED fixture file under paths at head (worktree when head is '')
    is checked; everything over a size cap is reported as a
    BASELINE-EXCEPTION, never a defect. Run
    `check_fixture_filesize.py <repo> - '' [paths...]` once to write down
    the pre-existing over-cap set. Malformed arguments exit 2 with a
    usage message, never a traceback.

Fixture recognition (two prongs):
  - directory convention: any path containing `/tests/fixtures/`
  - extension convention: any path ending `.safetensor` or `.safetensors`
    (tensor payloads are fixture material wherever they sit)

Size caps (exact constants, not vibes):
  - FILE-CAP  262144 B (256 kiB): any new fixture file above this is a
    defect (kind OVER-256K). No exceptions in diff mode; a file that was
    already tracked at base is by definition not added.
  - TEXT-TARGET 65536 B (64 kiB): a fixture file with a text extension
    (.md .txt .py .nim .yaml .yml, the audit trail and the check code)
    above this is a note (kind TEXT-64K), report-only, NOT a failure.
    Fixture data never rides a text extension: the json payloads sit in
    `.json.zst` frames, the tensor payloads in safetensors, both count
    as binary.
  - DIR-BUDGET 1572864 B (1.5 MiB): the model directory that contains the
    new file (first two path components under tests/fixtures/, one when
    the file sits directly under the fixture root) must stay under this
    total. Defect (kind DIR-1P5M) only when the ADDED files push the
    directory total past the size cap, i.e. the directory total was at or
    below the size cap at base and is above it at head. In sweep mode,
    directories already over the size cap are listed as BASELINE-EXCEPTION.

EXIT CODE: this script always exits 0. The caller greps the printed
summary line `defect kinds: {...}` for a non-empty dict, exactly like
scan_added.py. Do not wire it to a nonzero exit without changing the
house convention everywhere it is consumed.

CALLER CONTRACT: a crashed run (a git failure, a traceback) prints no
summary lines at all. A caller that reads `defect kinds:` must treat
the absence of the summary line as failure, never as pass.

Output lines (grep targets):
  <path>: MISSING-AT-HEAD: no size at head, the path was listed by the
    range but not sized (pathspec or case trap); counted as a defect,
    never silently as a zero-byte pass
  <path>: OVER-256K: <size> B (cap 262144)
  <path>: TEXT-64K: <size> B (target 65536, note only)
  <dir>: DIR-1P5M: <total> B at head vs <base-total> B at base
  <dir>: BASELINE-EXCEPTION: dir total <total> B (over 1572864 before this range)
  <path>: BASELINE-EXCEPTION: <size> B (over 262144 before this range)
  summary: fixture files checked: <n>
  summary: target notes: {kind: count}
  summary: baseline exceptions: {kind: count}
  summary: defect kinds: {kind: count}
  summary: TOTAL FIXTURE DEFECTS: <n>
"""
import os
import subprocess
import sys

FILE_CAP = 262144
TEXT_TARGET = 65536
DIR_TOTAL_CAP = 1572864
TENSOR_EXTS = (".safetensor", ".safetensors")
# Fixture data is blob material: .json payloads ride the .json.zst
# frame container, the binary treatment. The text extensions hold
# the audit trail and the check code only.
TEXT_EXTS = (".md", ".txt", ".py", ".nim", ".yaml", ".yml")

FIXTURE_MARK = "/tests/fixtures/"


def is_fixture(path):
    if FIXTURE_MARK in "/" + path:
        return True
    return path.endswith(TENSOR_EXTS)


def model_dir(path):
    """The model directory unit for the per-directory size cap.

    First two path components under tests/fixtures/ (one when the file
    sits directly under the fixture root). Extension-only fixture paths
    (tensor payloads outside tests/fixtures/) use their immediate parent.
    """
    if FIXTURE_MARK in path:
        after = path.split(FIXTURE_MARK, 1)[1].split("/")
        depth = min(2, len(after) - 1)
        return "/".join(after[:depth]) if depth else "(fixture root)"
    return "/".join(path.split("/")[:-1]) or "(repo root)"


def tracked_sizes(repo, rev, paths):
    """path -> size for every tracked file at rev (rev '' means worktree).

    Worktree mode uses stat so that unstaged/untracked-yet-staged state is
    visible; rev mode uses ls-tree -l blob sizes.
    """
    sizes = {}
    if rev:
        spec = [rev, "--"] + paths
        # core.quotePath=false keeps non-ASCII fixture names raw, matching
        # the worktree side, otherwise a quoted name misses every size lookup
        out = subprocess.run(
            ["git", "-c", "core.quotePath=false", "ls-tree", "-r", "-l", *spec],
            cwd=repo, capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if "\t" not in line:
                continue
            meta, name = line.split("\t", 1)
            parts = meta.split()
            if len(parts) == 4 and parts[1] == "blob":
                sizes[name] = int(parts[3])
        # ls-tree with a pathspec prefix prunes to that subtree, so keep
        # only the wanted paths when explicit paths were given
        if paths:
            wanted = set()
            for p in sizes:
                if any(p == q or p.startswith(q.rstrip("/") + "/") for q in paths):
                    wanted.add(p)
            sizes = {p: sizes[p] for p in wanted}
    else:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", *paths],
            cwd=repo, capture_output=True, check=True).stdout
        for name in out.decode("utf-8", "surrogateescape").split("\0"):
            if not name:
                continue
            try:
                sizes[name] = os.path.getsize(os.path.join(repo, name))
            except OSError:
                pass
        # Untracked files stay invisible to ls-files and to the worktree
        # diff, so a brand-new unstaged fixture escapes the size cap check
        # entirely. Sizes come from the working tree when the file sits
        # on disk.
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z",
             "--", *paths],
            cwd=repo, capture_output=True, check=True).stdout
        for name in out.decode("utf-8", "surrogateescape").split("\0"):
            if not name or name in sizes:
                continue
            try:
                sizes[name] = os.path.getsize(os.path.join(repo, name))
            except OSError:
                pass
    return sizes


def added_files(repo, base, head, paths):
    """Files ADDED by the range (present at head side, absent at base)."""
    spec = [f"{base}..{head}"] if head else [base]
    out = subprocess.run(
        ["git", "-c", "core.quotePath=false", "diff", "--diff-filter=A",
         "--name-only", *spec, "--", *paths],
        cwd=repo, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def dir_totals(sizes):
    """model_dir -> total bytes of fixture files inside it."""
    totals = {}
    for path, size in sizes.items():
        if is_fixture(path):
            d = model_dir(path)
            totals[d] = totals.get(d, 0) + size
    return totals


USAGE = (
    "Usage: check_fixture_filesize.py <repo> <base> <head-or-''> [paths...]\n"
    "  diff mode:  check_fixture_filesize.py <repo> <base-ref> <head-ref-or-''> [paths...]\n"
    "  sweep mode: check_fixture_filesize.py <repo> - <head-ref-or-''> [paths...]\n"
    "  ('-' is only ever the base argument; head is a ref or the empty"
    " string for the worktree)")


def main():
    if len(sys.argv) < 4:
        print(USAGE)
        return 2
    repo, base, head = sys.argv[1], sys.argv[2], sys.argv[3]
    paths = sys.argv[4:]
    if base == "-" and head == "-":
        # the sweep-mode head is a ref or '', never '-'; ls-tree would
        # reject the literal and crash the run with a traceback
        print(USAGE)
        return 2
    if head == "-":
        print(USAGE)
        return 2
    sweep = base == "-"
    missing = []
    if sweep:
        base_sizes = {}
        checked = tracked_sizes(repo, head, paths)
        files = sorted(checked)
    else:
        base_sizes = tracked_sizes(repo, base, paths)
        head_sizes = tracked_sizes(repo, head, paths)
        files = [p for p in added_files(repo, base, head, paths)]
        if not head:
            # Worktree mode: untracked new fixture files are invisible
            # to the diff and would escape every check. The untracked
            # sizes are already in head_sizes, the not-at-base test
            # identifies the new ones.
            files.extend(sorted(
                p for p in head_sizes
                if p not in base_sizes and p not in files and is_fixture(p)))
        checked = {}
        for p in files:
            if p not in head_sizes:
                # A listed path with no head size points at a pathspec
                # or case trap. A zero-byte default would pass every cap,
                # counting the path as a defect below instead.
                missing.append(p)
            else:
                checked[p] = head_sizes[p]
        full_head_sizes = head_sizes

    defects = {}
    notes = {}
    exceptions = {}

    def bump(d, kind):
        d[kind] = d.get(kind, 0) + 1

    for p in missing:
        print(f"{p}: MISSING-AT-HEAD: listed by the range but not sized "
              f"at head, counted as a defect")
        bump(defects, "MISSING-AT-HEAD")

    # per-file checks
    for path in files:
        if path not in checked:
            # missing-at-head paths already carry their defect line
            continue
        size = checked[path]
        if not is_fixture(path):
            continue
        if sweep:
            if size > FILE_CAP:
                print(f"{path}: BASELINE-EXCEPTION: {size} B "
                      f"(over {FILE_CAP} before this range)")
                bump(exceptions, "BASELINE-EXCEPTION")
        else:
            if size > FILE_CAP:
                print(f"{path}: OVER-256K: {size} B (cap {FILE_CAP})")
                bump(defects, "OVER-256K")
        if size > TEXT_TARGET and path.endswith(TEXT_EXTS):
            print(f"{path}: TEXT-64K: {size} B (target {TEXT_TARGET}, note only)")
            bump(notes, "TEXT-64K")

    # per-directory size cap check
    head_dirs = dir_totals(checked if sweep else full_head_sizes)
    base_dirs = dir_totals(base_sizes)
    for d in sorted(head_dirs):
        total = head_dirs[d]
        base_total = base_dirs.get(d, 0)
        if sweep:
            if total > DIR_TOTAL_CAP:
                print(f"{d}: BASELINE-EXCEPTION: dir total {total} B "
                      f"(over {DIR_TOTAL_CAP} before this range)")
                bump(exceptions, "BASELINE-EXCEPTION")
        else:
            if total > DIR_TOTAL_CAP and base_total <= DIR_TOTAL_CAP:
                print(f"{d}: DIR-1P5M: {total} B at head vs {base_total} B at base")
                bump(defects, "DIR-1P5M")

    n_defects = sum(defects.values())
    print(f"summary: fixture files checked: {len([f for f in files if is_fixture(f)])}")
    print(f"summary: target notes: {notes}")
    print(f"summary: baseline exceptions: {exceptions}")
    print(f"summary: defect kinds: {defects}")
    print(f"summary: TOTAL FIXTURE DEFECTS: {n_defects}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
