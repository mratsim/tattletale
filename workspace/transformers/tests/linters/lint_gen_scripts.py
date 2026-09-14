"""Fixture-generator linter, deterministic checks over testgen/gen_*.py.
Doc rules come from lint_docs.py, imported as the single source of truth.

Rule table (rule | trigger | severity):

| exactness-vocab | a comment says byte-exact, byte-identical, bit-exact, or bit-identical: a regen verifies by the instruments, bit equality = the codec payload contract only (noqa exempt) | counted |
| rule-id | trigger | severity |
|---|---|---|
| (lint_docs rules) | every doc rule of lint_docs.py over the script's docstrings and comments | per lint_docs.py |
| config-is-king | the script parses config.json and also hardcodes a geometry const (hidden size, head count, head or rotary dim, rank, top-k, layer count, intermediate dim, vocab) | counted |
| model-name-const | a module-level const holding a model checkpoint name | advisory |
| version-const | a module-level const holding a version string | advisory |
| main-guard | no `if __name__ == "__main__":` guard, or more than one | counted |
| dead-code | an unreachable `if False:` block, or a module-level def or class never referenced in the script | counted |
| commented-out-code | a whole-line comment carrying code text | counted |
| stale-run-path | a docstring run command naming a path that does not exist under the repo root | counted |
| rogue-file | a direct entry of a testgen/ directory that is neither a gen_*.py script nor an allowlisted helper | counted |


- ## docs serve API users, # comments serve maintainers and auditors
- write the contract, what the code does, its preconditions, its invariants
- every sentence stands readable to a fresh clone of the repository
- prefer bullets, tables, and diagrams over dense prose runs
- both skill files are the standard, read them first, they are always forgotten:
  - repo skill .agents/skills/writing-docs/ (SKILL.md, REFERENCE.md, EXAMPLES.md)
  - global skill ~/.pi/agent/skills/writing-code-doc/ (SKILL.md, references/REFERENCE.md)

The scan covers the gen_*.py scripts under the given paths, recursion included, non-generators out.

- run over a tree with `python3 lint_gen_scripts.py <files-or-dirs>...`
- autofix the mechanical doc classes with `python3 lint_gen_scripts.py --fix <files-or-dirs>...`
- the dogfood scan is `python3 lint_gen_scripts.py lint_gen_scripts.py`, it reports clean
- findings print one per line as `path:line: rule-id: reason`, sorted by path then line
- advisory findings print tagged [warning] and never set the exit code
- the lint pass never edits files, --fix runs the mechanical autofix first
- exit 0 means clean, --fix mode also asks for no mechanical doc finding left
- exit 1 on a counted finding or a mechanical doc finding left
- exit 2 on the missing lint_docs.py import
"""

import ast
import re
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
    # One source of truth for the doc rules:
    #   a missing lint_docs.py is
    # a hard error, never a silent skip of the doc checks.
    sys.stderr.write(
        "lint_gen_scripts: fatal: .agents/skills/writing-docs/tools/lint_docs.py "
        "not found above %s\n" % _FILE)
    raise SystemExit(2)
sys.path.insert(0, str(_SKILL_ROOT / ".agents" / "skills" / "writing-docs" / "tools"))
import lint_docs  # noqa, E402, the import sits after the sys.path setup


class Finding:
    """One linter finding:
      file path, line number, rule id, reason, advisory flag."""
    __slots__ = ("path", "line", "rule", "reason", "warning")

    def __init__(self, path, line, rule, reason, warning=False):
        self.path, self.line, self.rule = path, line, rule
        self.reason, self.warning = reason, warning


GEOMETRY_TOKENS = (
    "hidden", "head", "dim", "rank", "top_k", "topk", "intermediate",
    "inter_size", "ffn", "expert", "layer_count", "num_layers", "n_layers",
    "num_hidden_layers", "num_blocks", "vocab",
)
GEOMETRY_EXCLUDE_SUFFIX = ("_idx", "_index", "_file", "_dir", "_path", "_name")
GEOMETRY_EXCLUDE_TOKEN = "key"

MODEL_NAME_RE = re.compile(r"^Qwen")
VERSION_VALUE_RE = re.compile(r"^v?\d+(?:\.\d+){1,3}(?:[a-z].*)?$")
MAIN_GUARD_RE = re.compile(
    r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:")
DEAD_BRANCH_RE = re.compile(r"^(?:if|while)\s+(?:False|0)\s*:")
CODE_COMMENT_RE = re.compile(
    r"^(?:def |class |import |from |with .*:\s*$|try\s*:|except|@|print\(|"
    r"[A-Za-z_][\w.\[\]]*\s*(?:=|:=)\s*\S|(?:if|elif|else|for|while|"
    r"return|raise)\b.*:\s*$)")
SOURCE_SUFFIX_RE = re.compile(
    r"\.(?:py|md|nim|json|zst|sh|so|nims|cfg)$")

# Allowlist of the non-generator files a testgen/ directory may hold.
ROGUE_ALLOWLIST = ("FIXTURE_GENERATION.md", "fixture_stats.py")
RUN_CMD_RE = re.compile(r"^(?:\$|cd\s|\.?/|python\d?\s|uv\s|nim\s)")


def strip_comment(line):
    """Returns the line with a trailing # comment removed, keeping # inside
    string literals. Quote-counting, adequate for generator source lines."""
    in_str = None
    for i, ch in enumerate(line):
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "#":
            return line[:i]
    return line


def is_geometry_name(name):
    """Returns True for const names carrying model geometry.

    - the name holds a geometry token
    - the name is not an index or file reference
    - the name is not a checkpoint shard key count
    """
    low = name.lower()
    if low.endswith(GEOMETRY_EXCLUDE_SUFFIX):
        return False
    if GEOMETRY_EXCLUDE_TOKEN in low:
        return False
    return any(tok in low for tok in GEOMETRY_TOKENS)


def module_consts(tree):
    """Returns the module-level simple const assignments of a parsed module
    as a list of (line, name, value) triples.

    Only direct constants count, the targets, the augmented assigns, plus
    annotated assigns without a literal value stay out.
    """
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        line = node.lineno
        if isinstance(node, ast.AnnAssign):
            names = [node.target.id] if isinstance(node.target, ast.Name) else []
            value = node.value
        else:
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        if value is None or not isinstance(value, ast.Constant):
            continue
        for name in names:
            out.append((line, name, value.value))
    return out


def check_config_is_king(path, tree, code, findings):
    """Fires the counted config-is-king rule over one generator script.

    Expected input:
    - tree, the parsed module
    - code, the comment-stripped source text

    A script that parses config.json and also hardcodes a numeric geometry
    const reads its geometry from two places, the fixture then disagrees
    with the checkpoint the moment the config changes:
    - geometry belongs to the parsed config alone
    """
    if "config.json" not in code:
        return
    for line, name, value in module_consts(tree):
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and is_geometry_name(name):
            findings.append(Finding(
                path, line, "config-is-king",
                "const `%s = %s` hardcodes geometry in a script that parses "
                "config.json: take the value from the parsed config" % (name, value)))


def check_named_consts(path, tree, findings):
    """Fires the advisory const-name rules over one generator script.

    - model-name-const:
      a module const holding a model checkpoint name
    - version-const:
      a module const holding a version string

    Both belong in the config or the fixture manifest the script touches.
    """
    for line, name, value in module_consts(tree):
        if not isinstance(value, str):
            continue
        if MODEL_NAME_RE.match(value) or name.upper() in ("MODEL_NAME", "MODEL_ID"):
            findings.append(Finding(
                path, line, "model-name-const",
                "const `%s = %r` hardcodes a model checkpoint name"
                % (name, value), warning=True))
        elif VERSION_VALUE_RE.match(value) or "version" in name.lower():
            findings.append(Finding(
                path, line, "version-const",
                "const `%s = %r` hardcodes a version string"
                % (name, value), warning=True))


def check_main_guard(path, code_lines, findings):
    """Fires the counted main-guard rule, exactly one __main__ guard per
    generator script, so importing the module never triggers a recording."""
    guards = [i + 1 for i, ln in enumerate(code_lines)
              if MAIN_GUARD_RE.match(ln.strip())]
    if not guards:
        findings.append(Finding(
            path, 1, "main-guard",
            "no `if __name__ == \"__main__\":` guard: importing the module "
            "must never start a recording run"))
    for line in guards[1:]:
        findings.append(Finding(
            path, line, "main-guard", "second __main__ guard: keep one"))


def check_dead_code(path, tree, code_lines, findings):
    """Fires the counted dead-code rule over one generator script.

    - an unreachable `if False:` or `while False:` branch
    - a module-level def or class with no caller in the script
    """
    for i, ln in enumerate(code_lines, 1):
        if DEAD_BRANCH_RE.match(ln.strip()):
            findings.append(Finding(
                path, i, "dead-code",
                "unreachable `%s` branch: delete the block" % ln.strip()))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        if name == "main":
            continue
        uses = sum(1 for ln in code_lines
                   if re.search(r"\b%s\b" % re.escape(name), ln))
        if uses <= 1:
            findings.append(Finding(
                path, node.lineno, "dead-code",
                "`%s` is defined but never called in this script: delete it "
                "or move it beside its caller" % name))


BANNED_EXACTNESS_RE = re.compile(
    r"\bbyte[-_ ]exact\b|\bbyte[-_ ]identical\b|\bbit[-_ ]exact\b|\bbit[-_ ]identical\b",
    re.IGNORECASE)

def check_banned_exactness(path, lines, findings):
    """Fires the counted exactness-vocab rule over generator comment lines:
    byte/bit identity is banned as a pass criterion, a regen verifies by
    the instruments. The codec payload contract carries the ruled
    exception behind a noqa."""
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s.startswith("#") or "noqa" in s:
            continue
        body = s.lstrip("#").strip()
        if not body:
            continue
        m = BANNED_EXACTNESS_RE.search(body)
        if m:
            findings.append(Finding(
                path, i, "exactness-vocab",
                "banned exactness vocabulary '%s': bit/byte exactness violates the harness goal, robustness and portability across platforms; a regen verifies by the instruments" % m.group(0)))


def check_commented_out_code(path, lines, findings):
    """Fires the counted commented-out-code rule over the whole-line
    comments carrying code text, sentence-shaped prose stays exempt."""
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s.startswith("#") or s.startswith("#!") or s.startswith("# type:"):
            continue
        body = s.lstrip("#").strip()
        if not body or body.endswith(".") or body.startswith("noqa"):
            continue
        if CODE_COMMENT_RE.match(body):
            findings.append(Finding(
                path, i, "commented-out-code",
                "comment carries code text: delete it, the history keeps it"))


def check_stale_run_paths(path, doc_lines, findings):
    """Fires the counted stale-run-path rule over one generator script.

    Contract:
      the run commands in a docstring must be executable as written:
    - every file path a run line names exists under one plausible working
      directory (the repo root or an ancestor of the script itself)
    """
    bases = [_SKILL_ROOT]
    bases.extend(_FILE.parents)
    for line_no, text in doc_lines:
        if not RUN_CMD_RE.match(text) and not text.startswith(("Usage", "usage", "Run", "run")):
            continue
        for token in re.findall(r"[\w@.-]*(?:/[\w@.-]+)+", text):
            if "://" in token or not SOURCE_SUFFIX_RE.search(token):
                continue
            if any((base / token).exists() for base in bases):
                continue
            findings.append(Finding(
                path, line_no, "stale-run-path",
                "docstring names the path `%s`, no such file under the repo "
                "root" % token))


def check_rogue_files(path, findings):
    """Fires the counted rogue-file rule over every testgen/ directory under path.

    Expected input:
    - path, a scanned file or directory

    A testgen/ directory holds only generator scripts plus the allowlisted
    non-generator helpers in ROGUE_ALLOWLIST:
    - every direct entry of a testgen/ directory must be a gen_*.py script
      or an allowlisted filename, files and directories alike
    - __pycache__ and dotfile entries stay exempt:
      build and checkout artifacts, not debris
    - anything else is debris a retired script left behind, and the debris
      name in a tree a fresh clone checks out is the finding
    """
    pp = Path(path)
    if not pp.is_dir():
        return
    for d in sorted(pp.rglob("testgen")):
        if not d.is_dir():
            continue
        for entry in sorted(d.iterdir()):
            if entry.name in ROGUE_ALLOWLIST:
                continue
            if entry.name == "__pycache__" or entry.name.startswith("."):
                continue
            if entry.is_file() and entry.name.startswith("gen_") \
                    and entry.suffix == ".py":
                continue
            findings.append(Finding(
                entry, 1, "rogue-file",
                "`%s` sits in testgen/: only gen_*.py scripts and the "
                "allowlist %s belong there" % (entry.name, list(ROGUE_ALLOWLIST))))


def collect_files(paths):
    """Collects the gen_*.py scripts under paths, recursively.

    - sorted and deduped
    - only generators scan, the gen_ prefix is the scope marker
    """
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.rglob("gen_*.py")))
        elif pp.name.startswith("gen_") and pp.suffix == ".py":
            files.append(pp)
    out, seen = [], set()
    for f in sorted(set(files)):
        rf = f.resolve()
        if rf in seen:
            continue
        seen.add(rf)
        out.append(f)
    return out


def scan(path, text, findings):
    """Runs every rule over one generator script, appending to findings."""
    doc = lint_docs.lint_text(text, str(path))
    for fd in doc:
        findings.append(Finding(fd.path, fd.line, fd.rule, fd.reason, fd.warning))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        findings.append(Finding(
            path, 1, "dead-code", "file does not parse as Python"))
        return
    lines = text.splitlines()
    code_lines = [strip_comment(ln) for ln in lines]
    code = "\n".join(code_lines)
    doc_lines = [(n, c) for n, c, kind, _, _ in lint_docs.py_prose_lines(text)[0]
                 if kind == "doc"]
    check_config_is_king(path, tree, code, findings)
    check_named_consts(path, tree, findings)
    check_main_guard(path, code_lines, findings)
    check_dead_code(path, tree, code_lines, findings)
    check_commented_out_code(path, lines, findings)
    check_banned_exactness(path, lines, findings)
    check_stale_run_paths(path, doc_lines, findings)


def lint(paths):
    """Lints every collected generator script under paths.

    Returns the findings sorted by path, line, and rule id.
    """
    findings = []
    for f in collect_files(paths):
        scan(f, f.read_text(encoding="utf-8", errors="replace"), findings)
    for p in paths:
        check_rogue_files(p, findings)
    findings.sort(key=lambda x: (str(x.path), x.line, x.rule))
    return findings


def main(argv):
    """CLI entry point over a file and directory list, with --fix running
    the lint_docs mechanical autofix over the collected generators first.
    Returns the process exit code, 0 for clean, 1 for findings, 2 for failure."""
    args = argv[1:]
    fix_mode = "--fix" in args
    args = [a for a in args if a != "--fix"]
    mechanical_left = 0
    if fix_mode and args:
        mechanical_left = sum(lint_docs.fix_one_reported(f)
                              for f in collect_files(args))
    if not args:
        print(__doc__)
        return 2
    findings = lint(args)
    for fd in findings:
        tag = "warning" if fd.warning else "violation"
        print("%s:%d: %s: %s [%s]" % (fd.path, fd.line, fd.rule, fd.reason, tag))
    violations = sum(1 for fd in findings if not fd.warning)
    print("%d findings: %d violations, %d advisory warnings"
          % (len(findings), violations, len(findings) - violations))
    return 1 if (violations or mechanical_left) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
