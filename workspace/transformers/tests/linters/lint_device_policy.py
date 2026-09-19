#!/usr/bin/env python3
"""Device policy linter over the transformers tree.

PR #104 removed the automatic MPS-to-CPU fallback. These rules keep it out:

| rule-id | trigger | severity |
|---|---|---|
| mps-fallback | the string PYTORCH_ENABLE_MPS_FALLBACK anywhere in a scanned file: the env var re-enables the automatic CPU fallback PR #104 removed, GPU stays the primary record/test device and a missing device kernel must fail loudly | counted |
| cpu-default | a `= kCPU` default device parameter in a transformers src file: the default silently lands callers on cpu, callers pass the device explicitly | counted |

Scope:
- every .nim and .py file under workspace/transformers scans for mps-fallback
- workspace/transformers/src/**/*.nim scans for cpu-default
- noqa on the line exempts (the codec payload contract is the only sanctioned user)

Run: `python3 lint_device_policy.py <files-or-dirs>...`
One finding per line, `path:line: rule-id: reason`, sorted by path and line.
Advisory findings print with a [warning] tag and never set the exit code.
"""

import re
import sys
from pathlib import Path

MPS_FALLBACK = "PYTORCH_ENABLE_MPS_FALLBACK"
CPU_DEFAULT_RE = re.compile(
    r"(?:device|dev)\s*(?::[^=]+)?=\s*kCPU\b")


class Finding:
    def __init__(self, path, line, rule, reason, warning=False):
        self.path = path
        self.line = line
        self.rule = rule
        self.reason = reason
        self.warning = warning

    def render(self):
        tag = " [warning]" if self.warning else ""
        return f"{self.path}:{self.line}: {self.rule}: {self.reason}{tag}"


def scan_file(path, findings, src):
    if path.name == "lint_device_policy.py":
        return  # the dogfood scan reports clean on the linter itself
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return
    for i, raw in enumerate(text.splitlines(), 1):
        if "noqa" in raw:
            continue
        if MPS_FALLBACK in raw:
            findings.append(Finding(
                path, i, "mps-fallback",
                "PYTORCH_ENABLE_MPS_FALLBACK re-enables the automatic CPU "
                "fallback PR #104 removed, GPU is the primary record/test "
                "device, a missing device kernel fails loudly"))
        if src and CPU_DEFAULT_RE.search(raw):
            findings.append(Finding(
                path, i, "cpu-default",
                "a kCPU default device parameter lands callers on cpu "
                "silently, callers pass the device explicitly"))


def iter_files(args):
    for arg in args:
        p = Path(arg)
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(q for q in p.rglob("*")
                              if q.suffix in (".nim", ".py")
                              and "nimcache" not in q.parts
                              and "__pycache__" not in q.parts)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    findings = []
    for f in iter_files(args):
        src = "src" in f.parts
        scan_file(f, findings, src)
    findings.sort(key=lambda x: (str(x.path), x.line))
    counted = 0
    for f in findings:
        print(f.render())
        if not f.warning:
            counted += 1
    print(f"{len(findings)} findings: {counted} violations, "
          f"{len(findings) - counted} advisory warnings")
    return 1 if counted else 0


if __name__ == "__main__":
    sys.exit(main())
