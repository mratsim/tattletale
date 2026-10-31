"""C-family doc-comment linter for the writing-docs skill.

lint_md_nim_py.py stays the engine source of truth for the prose rules.
This linter applies the prose checks to C-family comment text
plus the c-family shapes in the table below.

Rule table (rule | severity | trigger):

| rule-id          | severity | trigger                                                                  |
| ---------------- | -------- | ------------------------------------------------------------------------ |
| block-closer     | counted  | a multi-line block closes with `*/` sharing its line with text           |
| param-list-style | counted  | a doxygen section breaks the aligned parameter-list layout               |
| semicolon        | counted  | a semicolon in comment prose (the engine rule over C text)               |
| em-dash          | counted  | an em-dash or en-dash in comment prose                                   |
| line-length      | counted  | a comment prose line over 140 characters                                 |
| article-eol      | counted  | a line ends on a dangling article or a stranded possessive               |
| stray-fragment   | counted  | a line ends on a bare connective or a fragment after the period          |
| colon-break      | counted  | a colon orphaned at line start or split from its lead phrase             |
| unit-split       | counted  | a line opens on a severed 1-word or 2-word continuation                  |
| severed-tail     | counted  | a continuing line ends on a 1-2 word tail after the last clause boundary |
| comma-stub       | counted  | a 1-2 word line ends on a comma, the clause belongs on the previous line |
| paren-split      | counted  | a line ends inside an open parenthesis                                   |
| quote-split      | counted  | a line ends inside a quoted string                                       |
| single-word-eol  | counted  | a 1-2 word stub line with reflow room on the previous line               |
| wrap-fits        | counted  | a statement or signature wraps although it fits in 140 columns           |
| brace-stmt       | counted  | a statement follows an opening brace on the same line                    |
| doc-marker       | counted  | a doc comment uses /* where the secp256k1 shape wants /**                |
| pointer-binding  | counted  | a pointer star binds to the name, the house rule binds it to the type    |
| comment-the      | counted  | a maintainer comment opens on the definite article                       |
| banned-vocab     | counted  | a blocklist word (the engine table)                                      |

Param-list layout, the secp256k1 shape:

    /** Serialize a pubkey object into a serialized byte sequence.
     *
     *  Parameters:
     *  Context: ctx:    pointer to a context object.
     *  Out:     out:    pointer to the output array.
     *  In:      input:  pointer to the input array.
     *
     *  Returns: 1 always.
     */

Usage:

    python3 lint_c_family.py <files-or-dirs>...

Exit codes: 0 clean, 1 violations found, 2 failure.
"""


import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lint_md_nim_py as engine  # noqa: E402  the engine is the source of truth

from lint_md_nim_py import (  # noqa: E402
    Finding,
    ARTICLE_EOL,
    CONNECTIVE_EOL,
    SINGLE_WORD_EOL_PREV_MAX,
    check_line,
    code_like,
    eol_token,
    pseudo_code,
    strip_backticks,
    structural,
    words,
)

C_SUFFIXES = (".c", ".h", ".cc", ".cpp", ".hpp", ".cxx")

# Doxygen section labels the parameter-list layout governs.
_SECTION_LABEL_RE = re.compile(
    r"^(\s*)(Returns|Parameters|Context|Out|In/Out|In):\s*(.*)$")
# A parameter entry inside a section: `name: description`.
_PARAM_ENTRY_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):(\s*)(\S.*)$")
# Declaration after a doc block, read to the statement end.
_TYPEISH_TOKENS = frozenset((
    "void", "char", "short", "int", "long", "float", "double",
    "bool", "unsigned", "signed", "const", "struct", "enum",
    "union", "static", "volatile",
))

_POINTER_DECL_RE = re.compile(
    r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)(\s+)(\*+)(?=\s*[A-Za-z_(\[])")
_POINTER_CAST_RE = re.compile(
    r"\(\s*(?:(?:unsigned|signed|const|struct|enum|union|long|short)\s+)*"
    r"([A-Za-z_][A-Za-z0-9_]*)(\s+)(\*+)\s*\)")
_POINTER_NOSPACE_RE = re.compile(
    r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)(\*+)(?=[A-Za-z_])")


def _is_typeish(token):
    """True for tokens that read only as a type in our code:
    builtin keywords, the tpc_ prefix, the stdint suffix."""
    return (token in _TYPEISH_TOKENS or token.startswith("tpc_")
            or token.endswith("_t"))


def _mask_code(text):
    """Blanks comment interiors and string/char literal contents,
    pointer rules read only code."""
    out = []
    i = 0
    n = len(text)
    in_block = False
    in_line = False
    in_str = False
    in_chr = False
    while i < n:
        ch = text[i]
        two = text[i:i + 2]
        if in_block:
            if two == "*/":
                in_block = False
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if in_line:
            if ch == "\n":
                in_line = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue
        if in_str or in_chr:
            if ch == "\\":
                out.append("  ")
                i += 2
                continue
            if (in_str and ch == '"') or (in_chr and ch == "'"):
                in_str = in_chr = False
                out.append(ch)
            else:
                out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if two == "/*":
            in_block = True
            out.append("  ")
            i += 2
            continue
        if two == "//":
            in_line = True
            out.append("  ")
            i += 2
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "'":
            in_chr = True
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def check_pointer_binding(path, text, findings):
    """Enforces star-to-type placement: the star belongs to the type,
    write type* name, never type *name.
    Covers declarations, parameters, return types, and casts.
    Multiplication survives, product left operands are never type tokens."""
    masked = _mask_code(text)
    lines = masked.splitlines()
    for no, line in enumerate(lines, 1):
        for m in _POINTER_CAST_RE.finditer(line):
            if _is_typeish(m.group(1)):
                findings.append(Finding(
                    path, no, "pointer-binding",
                    "the cast star binds to the type, write (%s*)" %
                    m.group(1)))
        for m in _POINTER_DECL_RE.finditer(line):
            if _is_typeish(m.group(1)):
                findings.append(Finding(
                    path, no, "pointer-binding",
                    "the star binds to the type, write %s* %s" %
                    (m.group(1), line[m.end():].split()[0]
                     if line[m.end():].split() else "name")))

        for m in _POINTER_NOSPACE_RE.finditer(line):
            if _is_typeish(m.group(1)):
                findings.append(Finding(
                    path, no, "pointer-binding",
                    "the star binds to the type, one space after, write "
                    "%s* %s" % (m.group(1),
                                line[m.end():].lstrip().split()[0]
                                if line[m.end():].lstrip().split()
                                else "name")))

_DECL_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def c_prose_lines(text):
    """Extracts comment prose from a C-family source file.

    Returns (entries, meta) with one entry per comment line.
    Meta adds "inline_closers", the multi-line blocks closing inline.

    `/** ... */` blocks carry kind `doc`, the API-facing comment.
    `/* ... */` and `// ...` carry kind `hash`, the maintainer comment.

    Decoration is stripped. Blank lines end the paragraph.
    URL schemes and string literals hide their comment markers.
    """
    out = []
    inline_closers = []
    in_block = False
    doc_block = False
    block_start = None
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if in_block:
            body = line
            end = body.find("*/")
            had_end = end >= 0
            if had_end:
                body = body[:end]
                in_block = False
            body = body.lstrip()
            if body.startswith("*"):
                body = body[1:]
                if body.startswith(" "):
                    body = body[1:]
            body = body.rstrip()
            kind = "doc" if doc_block else "hash"
            # a horizontal-rule banner is structure with the same
            # exemption the markdown extractor grants dash rules
            if re.match(r"^[-=─_*]{4,}$", body):
                continue
            if had_end and body and block_start is not None and block_start != i:
                # secp256k1 closes a multi-line block on its own line
                inline_closers.append(i)
            if not body and not had_end:
                # a blank comment line ends the paragraph
                out.append((i, "", kind, 0, False))
                continue
            out.append((i, body, kind, len(raw) - len(raw.lstrip()), False))
            continue
        # a line comment: the first `//` not part of a URL scheme and not
        # inside a string literal
        idx = None
        in_str = False
        j = 0
        while j < len(line) - 1:
            ch = line[j]
            if in_str:
                if ch == "\\":
                    j += 2
                    continue
                if ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "/" and line[j + 1] == "/" and \
                    (j == 0 or line[j - 1] != ":"):
                idx = j
                break
            j += 1
        if idx is not None:
            body = line[idx + 2:].strip()
            out.append((i, body, "hash",
                        len(raw) - len(raw.lstrip()), True))
            continue
        start = line.find("/*")
        if start >= 0:
            doc_block = line[start:].startswith("/**")
            body = line[start + 2:]
            end = body.find("*/")
            if end >= 0:
                body = body[:end].strip()
                if body.startswith("*"):
                    body = body[1:]
                    if body.startswith(" "):
                        body = body[1:]
                out.append((i, body, "doc" if doc_block else "hash",
                            len(raw) - len(raw.lstrip()), False))
                doc_block = False
            else:
                in_block = True
                block_start = i
                body = body.lstrip()
                if body.startswith("*"):
                    body = body[1:]
                    if body.startswith(" "):
                        body = body[1:]
                out.append((i, body, "doc" if doc_block else "hash",
                            len(raw) - len(raw.lstrip()), False))
    return out, {"inline_closers": inline_closers}


def _decl_is_void(source_lines, after_line):
    """Reads the declaration after a doc block. True when its return
    type is plain void."""
    joined = []
    for raw in source_lines[after_line:after_line + 12]:
        joined.append(raw.strip())
        chunk = " ".join(joined)
        if ";" in chunk or "{" in chunk:
            break
    decl = " ".join(joined)
    m = _DECL_NAME_RE.search(decl)
    if m is None:
        return False
    # Return type: the word before the function name, a void* return
    # reads as its own type
    prefix = decl[:m.start(1)].rstrip()
    if not prefix or prefix.endswith("*"):
        return False
    return re.split(r"[^A-Za-z0-9_]+", prefix)[-1] == "void"


_SUB_LABELS = frozenset(("Context", "Out", "In/Out", "In"))


def check_param_lists(path, blocks, source_lines, findings):
    """Fires the param-list-style rule over the doxygen blocks of one file.

    Layout:
    - a Parameters: opener line, then the Context, Out, In/Out, and In sub-sections
    - each sub-section opens with its first entry on the label line
    - entry names sit in one block-wide column, the entry descriptions
      align in one block-wide column, continuation lines sit on that column

    Context documents the session object, a void declaration gets no Returns: section.
    """
    for start, end, lines in blocks:
        for n, c, kind, indent, trailing in lines:
            if re.match(r"^\s*Args:", c):
                findings.append(Finding(
                    path, n, "param-list-style",
                    "the section label is Args, the session object gets "
                    "Context under a Parameters header"))
        has_opener = any(_SECTION_LABEL_RE.match(c)
                         and _SECTION_LABEL_RE.match(c).group(2) == "Parameters"
                         for n, c, *_ in lines)
        sub_lines = [(n, c) for n, c, *_ in lines
                     if _SECTION_LABEL_RE.match(c)
                     and _SECTION_LABEL_RE.match(c).group(2) in _SUB_LABELS]
        if sub_lines and not has_opener:
            findings.append(Finding(
                path, sub_lines[0][0], "param-list-style",
                "the parameter block opens with a Parameters: line, the "
                "Context, Out, In/Out, and In sub-sections follow it"))
        if has_opener and not sub_lines:
            on = [n for n, c, *_ in lines if _SECTION_LABEL_RE.match(c)
                  and _SECTION_LABEL_RE.match(c).group(2) == "Parameters"][0]
            findings.append(Finding(
                path, on, "param-list-style",
                "Parameters: opens nothing, the Context, Out, In/Out, "
                "and In sub-sections follow it"))
        # Name column: the block-wide constant 9, the widest sub-label pads there
        name_col = 9 if sub_lines else None
        shared = {"desc": None}
        sections = []
        cur = None
        for n, c, kind, indent, _trailing in lines:
            m = _SECTION_LABEL_RE.match(c)
            if m:
                label = m.group(2)
                content = m.group(3)
                gap = len(content) - len(content.lstrip()) if content \
                    else 0
                if content:
                    gap = len(m.group(0)) - len(m.group(1)) \
                        - len(label) - 1 - len(m.group(3))
                content_col = len(label) + 1 + gap
                if label == "Parameters":
                    if content.strip():
                        findings.append(Finding(
                            path, n, "param-list-style",
                            "Parameters: is a bare header, the "
                            "sub-sections follow on their own lines"))
                    cur = None
                    continue
                if content.strip():
                    if name_col is not None and content_col != name_col:
                        findings.append(Finding(
                            path, n, "param-list-style",
                            "the %s name column sits at %d, sub-labels "
                            "pad to one column at %d"
                            % (label, content_col, name_col)))
                    cur = {"label": label, "line": n, "col": content_col,
                           "desc_col": None, "entries": 0}
                    sections.append(cur)
                    pe = _PARAM_ENTRY_RE.match(content)
                    if pe:
                        desc_col = content_col + len(pe.group(2)) \
                            + 1 + len(pe.group(3))
                        cur["desc_col"] = desc_col
                        cur["entries"] += 1
                        if shared["desc"] is None:
                            shared["desc"] = desc_col
                        elif desc_col != shared["desc"]:
                            findings.append(Finding(
                                path, n, "param-list-style",
                                "the description column moved from %d to %d, "
                                "all entries align in one column"
                                % (shared["desc"], desc_col)))
                    elif not re.match(r"^[0-9]+:", content):
                        # free-text Returns content, not an entry list
                        cur["desc_col"] = "free"
                else:
                    findings.append(Finding(
                        path, n, "param-list-style",
                        "the %s section starts on the line after the "
                        "label, the label line carries the first entry" % label))
                    cur = {"label": label, "line": n, "col": None,
                           "desc_col": "free", "entries": 0}
                    sections.append(cur)
                continue
            if cur is None or not c.strip():
                continue
            if cur["col"] is None:
                # continuation of a bare Returns section
                cur["col"] = len(c) - len(c.lstrip())
                continue
            if cur["desc_col"] == "free":
                continue
            pe = _PARAM_ENTRY_RE.match(c)
            if pe:
                name_here = len(c) - len(c.lstrip())
                desc_col = name_here + len(pe.group(2)) + 1 + len(pe.group(3))
                if name_col is not None and name_here != name_col:
                    findings.append(Finding(
                        path, n, "param-list-style",
                        "the %s name column sits at %d, sub-labels pad "
                        "to one column at %d"
                        % (cur["label"], name_here, name_col)))
                elif shared["desc"] is None:
                    shared["desc"] = desc_col
                    cur["desc_col"] = desc_col
                elif desc_col != shared["desc"]:
                    findings.append(Finding(
                        path, n, "param-list-style",
                        "the description column moved from %d to %d, "
                        "all entries align in one column"
                        % (shared["desc"], desc_col)))
                cur["entries"] += 1
            else:
                cont_col = len(c) - len(c.lstrip())
                if shared["desc"] is not None and cont_col != shared["desc"]:
                    findings.append(Finding(
                        path, n, "param-list-style",
                        "a %s continuation line sits at column %d, the "
                        "section's column is %d"
                        % (cur["label"], cont_col, shared["desc"])))
        for sec in sections:
            if sec["label"] == "Context" and sec["entries"] > 1:
                findings.append(Finding(
                    path, sec["line"], "param-list-style",
                    "the Context section holds the session object only, "
                    "%d entries listed" % sec["entries"]))
        has_returns = any(l for l in lines
                          if _SECTION_LABEL_RE.match(l[1])
                          and _SECTION_LABEL_RE.match(l[1]).group(2) == "Returns"
                          and True)
        if has_returns and _decl_is_void(source_lines, end):
            findings.append(Finding(
                path, start, "param-list-style",
                "the documented declaration returns void, a Returns: "
                "section documents nothing"))


_FUNC_DECL_RE = re.compile(
    r"^(static\s+|inline\s+|_Thread_local\s+)*"
    r"(void|bool|char|float|double|long|unsigned|size_t|ssize_t|"
    r"u?int\d*_t?|tpc_\w+|struct\s+\w+|thrd_t)\b")


def _brace_depths(source_lines):
    """Running brace balance before each line, raw character count.

    - doc-example braces in comments balance out, the top-level
      boundary stays recognizable

    Returns: depths[i] = the brace balance before line i
    """
    depths = [0] * (len(source_lines) + 1)
    d = 0
    for i, line in enumerate(source_lines):
        depths[i] = d
        d += line.count("{") - line.count("}")
    depths[len(source_lines)] = d
    return depths


def _doc_marker_findings(path, source_lines, findings):
    """Doc-position comments must use the `/**` marker.

    - a `/* ... */` block at file scope whose next non-blank line opens
      a function definition, a function declaration or a type
      declaration documents that declaration
    - the secp256k1 convention reserves `/**` for doc comments
    - section banners, struct-field notes and function-interior notes
      stay `/*`

    Args:
      - path: lint target, used in the findings
      - source_lines: the file lines
      - findings: the findings list, appended in place
    """
    depths = _brace_depths(source_lines)
    i = 0
    n = len(source_lines)
    while i < n:
        stripped = source_lines[i].strip()
        if not (stripped.startswith("/*") and not stripped.startswith("/**")):
            i += 1
            continue
        j = i
        closer = -1
        while j < n:
            if "*/" in source_lines[j]:
                closer = j
                break
            j += 1
        if closer < 0:
            i += 1
            continue
        k = closer + 1
        while k < n and not source_lines[k].strip():
            k += 1
        if k < n and depths[i] == 0:
            nxt = source_lines[k].strip()
            if nxt.startswith("#"):
                # preprocessor sections take maintainer comments
                pass
            elif _FUNC_DECL_RE.match(nxt) and (
                    "(" in nxt or nxt.endswith("{") or nxt.endswith(";")):
                is_banner = False
                for b in range(i, closer + 1):
                    t = source_lines[b].strip()
                    if t.startswith("/*"):
                        t = t[2:]
                    if t.endswith("*/"):
                        t = t[:-2]
                    t = t.lstrip("*").strip()
                    if t and re.match(r"^[-=\u2500_*]{4,}$", t):
                        is_banner = True
                        break
                if not is_banner:
                    findings.append(Finding(
                        path, i + 1, "doc-marker",
                        "a comment documenting a function or type uses /*, "
                        "doc comments use /** (secp256k1 convention)"))
        i = closer + 1


_WRAP_OPEN_RE = re.compile(
    r"[,+(\*&|?]|>>|<<|&&|\|\||->|\.$")
_WRAP_CLOSE_RE = re.compile(r"[;)}]$")


def _line_comment_start(code):
    """Index of the trailing // line comment outside string literals, else None."""
    in_string = False
    i = 0
    while i < len(code):
        c = code[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            in_string = not in_string
        elif (c == "/" and not in_string and i + 1 < len(code)
                and code[i + 1] == "/"):
            return i
        i += 1
    return None


def _strip_c_strings(code):
    """Blanks string and char literal contents, keeps the quotes.

    - brace and paren analysis runs on the result
    - a brace inside a literal never reads as a block opener
    """
    out = []
    i = 0
    while i < len(code):
        c = code[i]
        if c in "\"'":
            quote = c
            out.append(quote)
            i += 1
            while i < len(code):
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == quote:
                    out.append(quote)
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _wrap_opens(stripped):
    """True when the line ends on a wrap continuation point.

    - a trailing comma continues a statement only inside an open parenthesis
    - an enum or initializer member ends on a comma at depth 0, one member per line is layout
    """
    if not _WRAP_OPEN_RE.search(stripped):
        return False
    if stripped.endswith(","):
        return stripped.count("(") > stripped.count(")")
    return True


def _wrap_fits_findings(path, source_lines, findings):
    """A wrapped line must not wrap when the joined form fits in 140 columns.

    - a statement line that ends on a comma inside an open parenthesis,
      on an operator or on an open parenthesis continues on the next line
    - a line that opens a block (`{` at the end) is a layout boundary,
      never a wrap candidate
    - the wrap closes when parentheses balance and the line ends on `;`, `)`, `}` or `{`
    - macro continuations (backslash), designated initializers and macro-body last lines are exempt

    Wrapping exists to fit the column budget, not to save scroll.
    Initializers and macro bodies keep one member per line.

    Args:
      - path: lint target, used in the findings
      - source_lines: the file lines
      - findings: the findings list, appended in place
    """
    in_block_comment = False
    prev_macro_cont = False
    n = 0
    while n < len(source_lines):
        line = source_lines[n]
        code = line
        if in_block_comment:
            if "*/" in code:
                code = code[code.index("*/") + 2:]
                in_block_comment = False
            else:
                n += 1
                continue
        if "/*" in code and "*/" not in code:
            in_block_comment = True
            code = code[:code.index("/*")]
        cut = _line_comment_start(code)
        if cut is not None:
            code = code[:cut]
        stripped = code.strip()
        is_macro_cont = stripped.endswith("\\")
        if (not stripped or stripped.startswith(("#", "*", "}", ".", "/"))
                or stripped.endswith("{")
                or is_macro_cont or prev_macro_cont):
            prev_macro_cont = is_macro_cont
            n += 1
            continue
        prev_macro_cont = False
        if _wrap_opens(stripped) and not _WRAP_CLOSE_RE.search(stripped):
            end = n + 1
            joined = stripped
            while end < len(source_lines):
                cont = source_lines[end].strip()
                if cont.startswith(("#", "//", "/*")):
                    break
                joined += " " + cont
                if (joined.count("(") - joined.count(")") <= 0
                        and cont.endswith((";", ")", "}", "{"))):
                    break
                end += 1
            if len(joined) <= 140 and "//" not in joined and "/*" not in joined:
                findings.append(Finding(
                    path, n + 1, "wrap-fits",
                    "the line wraps although the joined form fits in 140 columns, "
                    "join it"))
            n = end + 1
        else:
            n += 1


def _brace_stmt_findings(path, source_lines, findings):
    """A statement must not share its line with an unclosed opening brace.

    - code after a `{` still open at line end flags a jammed statement,
      an unclosed block opener stands alone, the statement goes below
    - closed one-line braces (`{...}` complete on the line) and macro
      continuation lines (trailing backslash) are the house layout

    Args:
      - path: lint target, used in the findings
      - source_lines: the file lines
      - findings: the findings list, appended in place
    """
    in_block_comment = False
    n = 0
    while n < len(source_lines):
        line = source_lines[n]
        code = line
        if in_block_comment:
            if "*/" in code:
                code = code[code.index("*/") + 2:]
                in_block_comment = False
            else:
                n += 1
                continue
        if "/*" in code and "*/" not in code:
            in_block_comment = True
            code = code[:code.index("/*")]
        cut = _line_comment_start(code)
        if cut is not None:
            code = code[:cut]
        stripped = code.strip()
        if (not stripped or stripped.startswith(("#", "*"))
                or stripped.endswith("\\")):
            n += 1
            continue
        scanned = _strip_c_strings(code)
        depth = 0
        last_open_level = None
        for c in scanned:
            if c == "{":
                last_open_level = depth
                depth += 1
            elif c == "}":
                depth -= 1
        if (last_open_level is not None and depth > last_open_level):
            tail = scanned[scanned.rindex("{") + 1:].strip()
            if tail:
                findings.append(Finding(
                    path, n + 1, "brace-stmt",
                    "a statement follows the opening brace on the same line, "
                    "put the statement on its own line"))
        n += 1


def lint_c_family(path, text, findings):
    """Runs every rule over one C-family file's comments."""
    entries, meta = c_prose_lines(text)
    source_lines = text.splitlines()
    for no in meta["inline_closers"]:
        findings.append(Finding(
            path, no, "block-closer",
            "the block closer shares its line with comment text, close "
            "the block with `*/` on its own line"))
    _doc_marker_findings(path, source_lines, findings)
    _wrap_fits_findings(path, source_lines, findings)
    _brace_stmt_findings(path, source_lines, findings)

    # group the entries into contiguous blocks for the param-list pass
    blocks = []
    cur = []
    prev_no = None
    for entry in entries:
        if prev_no is not None and entry[0] != prev_no + 1:
            if cur:
                blocks.append(cur)
            cur = []
        cur.append(entry)
        prev_no = entry[0]
    if cur:
        blocks.append(cur)

    doc_blocks = []
    for block in blocks:
        kinds = {e[2] for e in block}
        if "doc" in kinds:
            first = block[0][0]
            last = block[-1][0]
            # a /** block opens the block, keep its line span
            if any(e[2] == "doc" for e in block):
                doc_blocks.append((first, last, block))

    check_param_lists(path, doc_blocks, source_lines, findings)
    check_pointer_binding(path, text, findings)

    blank_nos = {e[0] for e in entries if not e[1]}
    prev_prose = None
    prev_text = {}
    prev_kind = {}
    for n, c, kind, indent, trailing in entries:
        prev_text[n] = c
        prev_kind[n] = kind
    for n, c, kind, indent, trailing in entries:
        if not c:
            prev_prose = None
            continue
        # an indented code sample inside a comment is code-like
        # because the engine test is anchored. Strip the indentation first.
        # A statement shape deep inside a comment is example code
        if code_like(c.lstrip()) or (
                len(c) - len(c.lstrip()) >= 4
                and c.rstrip().endswith((";", "{", "}"))):
            prev_prose = None
            continue
        check_line(path, n, c, kind, False, prev_text, prev_kind, findings)
        if structural(c) or code_like(c):
            prev_prose = None
            continue
        wc = len(words(strip_backticks(c)))
        stub_ok = (wc <= 2 and "`" not in c and not c.endswith(":")
                   and not pseudo_code(c) and indent < 3
                   and not c[0:1].isupper()
                   and (n + 1) not in blank_nos
                   and prev_prose is not None
                   and prev_prose[0] + 1 == n
                   and prev_prose[2] == kind
                   and len(prev_prose[1]) < SINGLE_WORD_EOL_PREV_MAX)
        if stub_ok:
            findings.append(Finding(
                path, n, "single-word-eol",
                "line ends on a %d-word stub, the previous line had room "
                "to reflow" % wc))
        if wc <= 2 and c.endswith(","):
            findings.append(Finding(
                path, n, "comma-stub",
                "a %d-word line ends on a comma, the clause belongs "
                "on the previous line" % wc))
        if (prev_prose is not None and prev_prose[2] == kind
                and prev_prose[0] + 1 == n and c[:1].islower()
                and not trailing and not prev_prose[3]):
            prev_txt = re.sub(r"`[^`]*`", " XcodeX ",
                              prev_prose[1].rstrip())
            tail = re.split(r"[,;:.]", prev_txt)[-1].strip()
            twc = len(words(tail))
            if 1 <= twc <= 2 and not tail.endswith("etc"):
                findings.append(Finding(
                    path, prev_prose[0], "severed-tail",
                    "line ends on a %d-word tail after the last clause "
                    "boundary, the sentence continues" % twc))
        prev_prose = (n, c, kind, trailing)


def main(argv):
    """CLI entry point over files and directories. Returns the process
    exit code with 0 for clean, 1 for violations, 2 for failure."""
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    files = []
    for p in args:
        pp = os.path.abspath(p)
        if not os.path.exists(pp):
            sys.stderr.write("lint_c_family: missing path: %s\n" % p)
            return 2
        if os.path.isdir(pp):
            for root, _dirs, names in os.walk(pp):
                for name in sorted(names):
                    if name.endswith(C_SUFFIXES):
                        files.append(os.path.join(root, name))
        elif pp.endswith(C_SUFFIXES):
            files.append(pp)
    files.sort()
    findings = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        lint_c_family(os.path.relpath(f), text, findings)
    findings.sort(key=lambda x: (str(x.path), x.line, x.rule))
    violations = [fd for fd in findings if not fd.warning]
    for fd in findings:
        print("%s:%d: %s: %s [%s]" % (fd.path, fd.line, fd.rule, fd.reason,
                                      "warning" if fd.warning else "violation"))
    print("%d findings: %d violations, %d advisory warnings"
          % (len(findings), len(violations), len(findings) - len(violations)))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
