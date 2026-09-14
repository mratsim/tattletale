#!/usr/bin/env python3
"""Fixture-tree linter, deterministic checks over the fixture trees.
The finding record shape comes from lint_docs.py, the single source of truth for the linter family.

Rule table (rule | trigger | severity):

| rule-id | trigger | severity |
|---|---|---|
| file-cap | a fixture file over 262144 B outside the recorded exceeder families | counted |
| dir-cap | one fixture family model dir over 1.5 MiB outside the recorded exceeder families | counted |
| fixture-tier | a family dir name outside the tier naming rule | counted |
| record-two-frame | a `.descriptors.json.zst` sidecar in the two-frame record format | counted |
| sidecar-schema | a `.stats.json.zst` frame failing the uniform record shape | counted |

- the exceeder carve-out (ALLOWED_EXCEEDERS below) is a recorded operator
  decision on the size caps, exactly the two listed families may exceed
  the caps, no other file or family may join the list silently

Golden rules:
- audience, ## docs serve API users, # comments serve maintainers and auditors
- write the contract, what the code does, its preconditions, its invariants, never the journey
- no hidden context, every sentence stands readable to a fresh clone of the repository
- walls are hostile, prefer bullets, tables, and diagrams over dense prose runs
- both skill files are the standard, read them first, they are always forgotten
  - the repo skill .agents/skills/writing-docs/ (SKILL.md, REFERENCE.md, EXAMPLES.md)
  - the global skill ~/.pi/agent/skills/writing-code-doc/ (SKILL.md, references/REFERENCE.md)

- scanned under the fixture root, every regular file (size caps, sidecar schema)
- scanned under the fixture root, every first-level family directory (tier naming rule, dir cap)
- a file argument scans itself, the family-scoped rules need the tree view
- the hf_models contents are machine-local (gitignored)
- real model downloads, symlinks into a local store, or an empty dir
  all stand there, nothing is mandated or policed
- the uniform record shape, one entry per tensor, validated by sidecar-schema:
  - top level, `schema` = the uniform record id, nonempty `source`, a `tensors` object
  - fixed probabilities, the 11 quantile keys, min and max plus p01 to p99, hex f32 bit patterns
  - binade-log histogram keys, sparse `key:count` pairs, keys uint16, sorted, reserved bins inside
  - band inputs, `mean_abs`, `signed_mean`, `tail_probability` hex f64, `tail_edge` nonnegative int
- frames decompress through the system zstd tool (the harness system-dynlib default)
- a missing zstd binary is a hard error
- running over a tree takes python3 lint_fixtures.py <files-or-dirs>...
- the dogfood scan is python3 lint_fixtures.py lint_fixtures.py, it reports clean
- output is one finding per line, `path:0: rule-id: reason` for tree-level findings
- the linter is a reporting tool, it never edits files
- exit 0 means clean
- exit 1 means at least one counted finding
- exit 2 means the lint_docs.py import or the zstd binary is missing
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

_FILE = Path(__file__).resolve()


def _skill_root():
    """Returns the repo root holding .agents/skills/writing-docs/tools,
    walking up from this file, or None."""
    d = _FILE.parent
    while d.parent != d:
        if (d / ".agents" / "skills" / "writing-docs" / "tools" / "lint_docs.py").is_file():
            return d
        d = d.parent
    return None


_SKILL_ROOT = _skill_root()
if _SKILL_ROOT is None:
    # One source of truth for the finding record shape:
    # a missing lint_docs.py import is a hard error, never a silent skip.
    sys.stderr.write(
        "lint_fixtures: fatal: .agents/skills/writing-docs/tools/lint_docs.py "
        "not found above %s\n" % _FILE)
    raise SystemExit(2)
sys.path.insert(0, str(_SKILL_ROOT / ".agents" / "skills" / "writing-docs" / "tools"))
import lint_docs  # noqa  # the sys.path setup runs above, E402 silent

Finding = lint_docs.Finding

FILE_CAP = 262144
DIR_CAP = 1572864

UNIFORM_SCHEMA_ID = "ttt-tf-004-uniform-stats"
QUANTILE_KEYS = ("min", "max", "p01", "p05", "p10", "p25", "p50", "p75",
                 "p90", "p95", "p99")

FAMILY_RE = re.compile(r"^(bf16|exl3)-(\d{2})-[a-z0-9-]+$")
ZERO_BIN = 65534
SUBNORMAL_BIN = 65535

# The recorded exceeder carve-out, exactly these two fixture families may
# exceed the size caps. No other entry joins the list silently, a new
# exception needs a recorded operator decision.
ALLOWED_EXCEEDERS = ("exl3-00-codec", "exl3-01-block-02-trace")

HEX_F32_RE = re.compile(r"^(?:0x[0-9A-Fa-f]{8}|-inf)$")
HEX_F64_RE = re.compile(r"^0x[0-9A-Fa-f]{16}$")


def decompress(path):
    """Returns the JSON text of one zstd frame, or None when the frame does
    not decompress. Raises SystemExit when the zstd binary is missing."""
    try:
        proc = subprocess.run(
            ["zstd", "-dc", "-q", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        sys.stderr.write("lint_fixtures: fatal: the zstd tool is missing, "
                         "the frames cannot be validated\n")
        raise SystemExit(2)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def check_quantiles(tensor, errors):
    """Appends the quantile errors of one tensor entry.

    The fixed probability set is the hex f32 bit patterns, nothing else.
    """
    q = tensor.get("quantiles")
    if not isinstance(q, dict):
        errors.append("quantiles object missing")
        return
    keys = tuple(q.keys())
    if keys != QUANTILE_KEYS and set(keys) == set(QUANTILE_KEYS):
        errors.append("quantile keys off the stored order: %s" % (keys,))
    elif set(keys) != set(QUANTILE_KEYS):
        errors.append("quantile keys off the fixed probability set: %s"
                      % (sorted(set(keys) ^ set(QUANTILE_KEYS)),))
    for k in keys:
        v = q[k]
        if not isinstance(v, str) or not HEX_F32_RE.match(v.strip()):
            errors.append("quantile `%s` is not a hex f32 bit pattern" % k)


def check_histogram(tensor, errors):
    """Appends the histogram errors of one tensor entry, the sparse uint16 keys
    in sorted order, the reserved zero and subnormal bins inside the range."""
    h = tensor.get("histogram")
    if h is None:
        return
    if not isinstance(h, dict):
        errors.append("histogram is not an object")
        return
    total = h.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        errors.append("histogram total missing or negative")
    buckets = h.get("buckets")
    if not isinstance(buckets, str):
        errors.append("histogram buckets missing")
        return
    last = -1
    for pair in buckets.split(","):
        if ":" not in pair:
            errors.append("histogram pair `%s` carries no key:count" % pair)
            continue
        k, _sep, _cnt = pair.partition(":")
        try:
            key = int(k)
        except ValueError:
            errors.append("histogram key `%s` is not an integer" % k)
            continue
        if key < 0 or key > SUBNORMAL_BIN:
            errors.append("histogram key `%d` outside the uint16 range" % key)
        elif key <= last and last >= 0:
            errors.append("histogram keys out of sorted order at `%d`" % key)
        last = max(last, key)


BAND_KEYS = ("mean_abs", "signed_mean", "tail_probability")


def check_band_and_fields(tensor, errors):
    """Appends the band-input and field errors of one tensor entry.

    - a fingerprint-only entry carries no band input
    - a descriptor-carried entry carries the hex f64 band inputs and a
      nonnegative `tail_edge`
    """
    has_band = any(tensor.get(k) for k in BAND_KEYS)
    if not has_band:
        return
    for key in BAND_KEYS:
        v = tensor.get(key)
        if not isinstance(v, str) or not HEX_F64_RE.match(v):
            errors.append("`%s` is not a hex f64 bit pattern" % key)
    tail_edge = tensor.get("tail_edge")
    if not isinstance(tail_edge, int) or isinstance(tail_edge, bool) \
            or tail_edge < 0:
        errors.append("`tail_edge` missing or negative")


def check_uniform_record(path, text, findings):
    """Validates one decompressed stats frame against the uniform record shape,
    appending one finding per defect with the tensor named."""
    try:
        doc = json.loads(text)
    except ValueError:
        findings.append(Finding(
            path, 0, "sidecar-schema", "frame payload is not valid JSON"))
        return
    if doc.get("schema") != UNIFORM_SCHEMA_ID:
        findings.append(Finding(
            path, 0, "sidecar-schema",
            "schema id `%s` is not the uniform record id `%s`"
            % (doc.get("schema"), UNIFORM_SCHEMA_ID)))
    if not doc.get("source"):
        findings.append(Finding(
            path, 0, "sidecar-schema", "`source` key missing or empty"))
    tensors = doc.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        findings.append(Finding(
            path, 0, "sidecar-schema", "`tensors` object missing or empty"))
        return
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            findings.append(Finding(
                path, 0, "sidecar-schema",
                "tensor `%s` is not an object" % name))
            continue
        errors = []
        check_quantiles(tensor, errors)
        check_histogram(tensor, errors)
        check_band_and_fields(tensor, errors)
        for e in errors:
            findings.append(Finding(
                path, 0, "sidecar-schema", "tensor `%s`: %s" % (name, e)))


def family_of(path, root):
    """Returns the family name of a path under the fixture root, the first
    path component below the root, or None when the path sits at the root."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def scan_fixtures(root, findings):
    """Runs the fixture rules over one fixtures root.

    - the size caps, per file and per model dir (a family groups its
      model dirs one level down, a family without model subdirs is its
      own single model unit)
    - the tier naming rule, several families may share one stage number
    - the record sidecars
    """
    families = sorted(d for d in root.iterdir() if d.is_dir())
    for family in families:
        if not FAMILY_RE.match(family.name):
            findings.append(Finding(
                family, 0, "fixture-tier",
                "family dir `%s` outside the tier naming rule bf16-NN-<slug> / "
                "exl3-NN-<slug>" % family.name))
        exempt = family.name in ALLOWED_EXCEEDERS
        groups = {}
        for child in sorted(family.iterdir()):
            key = child if child.is_dir() else family
            groups.setdefault(key, 0)
        for p in sorted(family.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            key = p.parent
            while key.parent != family and key != family:
                key = key.parent
            size = p.stat().st_size
            groups[key] = groups.get(key, 0) + size
            if size > FILE_CAP and not exempt:
                findings.append(Finding(
                    p, 0, "file-cap",
                    "file is %d B, cap is %d B" % (size, FILE_CAP)))
            if p.name.endswith(".descriptors.json.zst"):
                findings.append(Finding(
                    p, 0, "record-two-frame",
                    "two-frame record format: regenerate as the uniform "
                    "stats frame"))
            elif p.name.endswith(".stats.json.zst"):
                text = decompress(p)
                if text is None:
                    findings.append(Finding(
                        p, 0, "sidecar-schema", "frame does not decompress"))
                else:
                    check_uniform_record(p, text, findings)
        for group, total in sorted(groups.items(), key=lambda kv: kv[0].name):
            if total > DIR_CAP and not exempt:
                findings.append(Finding(
                    group, 0, "dir-cap",
                    "model dir total is %d B, cap is %d B" % (total, DIR_CAP)))


def locate_roots(path):
    """Returns the (kind, root) pairs for one given path.

    - a directory named fixtures or hf_models is the root itself
    - a directory holding both a fixtures and an hf_models child maps to each root
    - a file maps to the nearest ancestor named fixtures or hf_models
    - an unresolvable path maps to nothing, the caller notes the skip
    """
    pp = Path(path).resolve()
    out = []
    if pp.is_dir():
        if pp.name in ("fixtures", "hf_models"):
            out.append((pp.name, pp))
        for child in ("fixtures", "hf_models"):
            if (pp / child).is_dir():
                out.append((child, pp / child))
        return out
    for anc in pp.parents:
        if anc.name in ("fixtures", "hf_models"):
            out.append((anc.name, anc))
            break
    return out


def scan_file(path, findings):
    """Runs the file-scoped fixture rules over one fixture file.

    A file argument scans itself, so the sidecar record rules fire on the one file.
    The family-scoped rules (size caps, tier naming) need the directory view.

    - Args:
      the resolved path and the findings list
    """
    if path.name.endswith(".stats.json.zst"):
        text = decompress(path)
        if text is None:
            findings.append(Finding(
                path, 0, "sidecar-schema", "frame does not decompress"))
        else:
            check_uniform_record(path, text, findings)
    elif path.name.endswith(".descriptors.json.zst"):
        findings.append(Finding(
            path, 0, "record-two-frame",
            "two-frame record format: regenerate as the uniform "
            "stats frame"))


def lint(paths):
    """Lints every fixture or hf_models root under paths.

    Returns the findings sorted by path, rule, and reason.
    """
    findings = []
    for p in paths:
        pp = Path(p).resolve()
        roots = locate_roots(p)
        if not roots:
            sys.stderr.write("lint_fixtures: no fixtures or hf_models root "
                             "found for %s, skipping\n" % p)
            continue
        if pp.is_file():
            # A file argument scans itself, the family-scoped rules
            # (size caps, tier naming) need the tree view.
            scan_file(pp, findings)
            continue
        for kind, root in roots:
            if kind == "fixtures":
                scan_fixtures(root, findings)
            # hf_models roots map to nothing, the contents are machine-local
    findings.sort(key=lambda x: (str(x.path), x.rule, x.reason))
    return findings


def main(argv):
    """CLI entry point, a file and directory list.
    Returns the process exit code (0 clean, 1 findings, 2 failure)."""
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    findings = lint(args)
    for fd in findings:
        print("%s:%d: %s: %s" % (fd.path, fd.line, fd.rule, fd.reason))
    print("%d findings" % len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
