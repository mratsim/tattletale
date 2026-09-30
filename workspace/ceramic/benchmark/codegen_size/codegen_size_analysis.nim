## Codegen-size reporting for the codesize runners, bencher reports.nim style.
##
## Static analysis of emitted Metal Shading Language. One measurement covers
## a `const x = metal:` block expansion, structs and prototypes included.
## Counting rules, applied brace-balanced over the source lines:
##
## | field         | counting rule                                        |
## | ------------- | ---------------------------------------------------- |
## | types         | file-scope struct definitions, bodies skipped        |
## | funcs         | function definitions, prototypes excluded            |
## | vars          | local declarations inside function bodies            |
## | function LOC  | body lines between header line and closing brace     |
## | overload fams | functions grouped by un-mangled base name            |
## | calls         | call-position symbol occurrences, protos and structs |
## |               | excluded, address-taken symbols are invisible        |
## | ovl           | mangling variants per Nim origin name                |
## | inl/ninl      | `inline` qualifier vs kernel and bare helper headers |
##
## vars never counts struct fields, fields belong to the type declaration.
## Scanner input is read-only, MSL is never transformed.
import std/[algorithm, strformat, strutils, tables]

type
  CgsStats* = object
    bytes*: int         ## MSL source size in bytes
    lines*: int         ## physical lines in the MSL source
    types*: int         ## file-scope struct definitions
    vars*: int          ## local declarations inside function bodies
    funcs*: int         ## function definitions, prototypes excluded
    inlineFuncs*: int   ## definitions with the `inline` qualifier
    nonInlineFuncs*: int ## kernels and bare helper definitions
    calls*: int         ## static call sites over all defined functions

  CgsLocBuckets* = object
    le5*: int           ## functions of at most 5 body lines
    le30*: int          ## functions of at most 30 body lines
    le70*: int          ## functions of at most 70 body lines
    le150*: int         ## functions of at most 150 body lines
    le300*: int         ## functions of at most 300 body lines
    le700*: int         ## functions of at most 700 body lines
    le1400*: int        ## functions of at most 1400 body lines
    gt1400*: int        ## functions above 1400 body lines

  CgsOverloadBuckets* = object
    groups*: CgsLocBuckets  ## overload families bucketed by summed member LOC
    members*: CgsLocBuckets ## individual functions of the bucketed families
    families*: int          ## overload families, 1-member families included

  CgsOwnerKind* = enum
    okFunction       ## kernel or device helper definition
    okStruct         ## file-scope struct definition
    okGlobal         ## file-scope variable declaration
    okPreamble       ## includes and blanks outside declarations

  CgsOwner* = object
    symbol*: string     ## emitted Metal name, mangled spelling
    origin*: string     ## Nim origin name, mangling suffix stripped
    kind*: CgsOwnerKind
    loc*: int           ## attributed lines, header, protos, and body included
    bytes*: int         ## attributed characters, line breaks included
    calls*: int         ## static call-site count over the whole source
    variants*: int      ## mangling variants of the origin, 0 for non-functions

  CgsAttribution* = object
    owners*: seq[CgsOwner]      ## sorted by loc desc
    totalLines*: int            ## all lines of the MSL source
    totalBytes*: int            ## all bytes of the MSL source
    families*: int              ## overload groups over Nim origin names
    largestFamily*: int         ## variant count of the largest group
    largestFamilyName*: string  ## Nim origin of the largest group
    singleVariantFamilies*: int ## groups with exactly one variant
    pareto*: seq[string]        ## Nim origins of the minimal top set
    paretoLines*: int           ## lines of that set
    paretoPct*: float           ## that set's share of totalLines, percent

  CgsReceipt* = tuple[name: string, msl: string]

const
  identStart = {'A'..'Z', 'a'..'z', '_'}
  identChars = {'A'..'Z', 'a'..'z', '0'..'9', '_'}

type
  FuncInfo = object
    name: string     ## mangled name as emitted
    base: string     ## un-mangled family key, whole name when unmangled
    isInline: bool   ## definition header opens on `inline`
    loc: int         ## body lines between header line and closing brace
    vars: int        ## local declarations in the body
    calls: int       ## static call sites of the mangled symbol
    headerLine: int  ## line index of the definition header

  Span = tuple[a, b: int]  ## inclusive line range

  PendingProto = object
    symbol: string   ## mangled symbol of the awaited definition
    span: Span       ## prototype statement lines

  OwnerInfo = object
    symbol: string   ## emitted Metal name, mangled spelling
    origin: string   ## Nim origin name, mangling suffix stripped
    kind: CgsOwnerKind
    spans: seq[Span] ## attributed line ranges, exactly one owner per line

  Scan = object
    stats: CgsStats
    funcs: seq[FuncInfo]
    owners: seq[OwnerInfo]
    pendingProtos: seq[PendingProto] ## prototypes awaiting their definition
    protoSpans: seq[Span]    ## prototype line ranges
    rawLines: seq[string]
    codeLines: seq[string]   ## comment- and literal-blanked lines
    endsWithNewline: bool

proc blankComments(line: string, inBlock: var bool): string =
  ## Copy of line with comment text and string or char literal contents
  ## blanked to spaces, offsets preserved. inBlock carries block-comment
  ## state across the lines of one source.
  result = line
  var i = 0
  while i < line.len:
    if inBlock:
      result[i] = ' '
      if line[i] == '*' and i + 1 < line.len and line[i + 1] == '/':
        inBlock = false
        result[i] = ' '
        result[i + 1] = ' '
        inc i, 2
        continue
      inc i
      continue
    case line[i]
    of '/':
      if i + 1 < line.len and line[i + 1] == '/':
        while i < line.len:
          result[i] = ' '
          inc i
        continue
      if i + 1 < line.len and line[i + 1] == '*':
        inBlock = true
        result[i] = ' '
        result[i + 1] = ' '
        inc i, 2
        continue
      inc i
    of '"', '\'':
      let q = line[i]
      result[i] = ' '
      inc i
      while i < line.len:
        if line[i] == '\\':
          if i + 1 < line.len:
            result[i] = ' '
            result[i + 1] = ' '
          inc i, 2
        elif line[i] == q:
          result[i] = ' '
          inc i
          break
        else:
          result[i] = ' '
          inc i
    else:
      inc i

proc braceDelta(line: string): int =
  ## Net brace depth change over one line, comments and literals blanked.
  var inBlock = false
  let c = blankComments(line, inBlock)
  for ch in c:
    if ch == '{':
      inc result
    elif ch == '}':
      dec result

proc scanHeader(line: string): tuple[term: char, idx: int, paren: int] =
  ## Returns the first `;` or `{` outside comments and literals.
  ## paren carries the parenthesis depth at line end.
  ## term is ' ' when neither terminator occurs on the line.
  var inBlock = false
  let c = blankComments(line, inBlock)
  var paren = 0
  result = (' ', -1, 0)
  var i = 0
  while i < c.len:
    case c[i]
    of '(':
      inc paren
      inc i
    of ')':
      dec paren
      inc i
    of ';':
      if paren == 0:
        return (';', i, paren)
      inc i
    of '{':
      if paren == 0:
        return ('{', i, paren)
      inc i
    else:
      inc i
  result.paren = paren

proc headerName(header: string): string =
  ## Returns the function name as emitted.
  ## That name is the character run before the first open paren of the header.
  ## Operator spellings keep their symbols, as in `===___<hash>`.
  let p = header.find('(')
  if p <= 0:
    return ""
  const nameTail = identChars + {'=', '<', '>', '!', '+', '-', '*', '%', '&', '|', '^', '~'}
  var e = p - 1
  while e >= 0 and header[e] in nameTail:
    dec e
  result = header[e + 1 .. p - 1]

proc isHashTail(s: string): bool =
  ## Returns true when s is a `^[0-9A-Za-z]+(___[0-9A-Za-z]+)*$` shaped run.
  var i = 0
  var segs = 0
  while i < s.len:
    var j = i
    while j < s.len and s[j] in {'A'..'Z', 'a'..'z', '0'..'9'}:
      inc j
    if j == i:
      return false
    inc segs
    i = j
    if i < s.len:
      if i + 2 < s.len and s[i] == '_' and s[i + 1] == '_' and s[i + 2] == '_':
        inc i, 3
      else:
        return false
  result = segs > 0

proc unMangled(name: string): string =
  ## Returns the overload family key of name.
  ## A mangling suffix is base62 text between `___` separators.
  ## A name without a mangling suffix is its own key.
  var i = name.find("___")
  while i > 0:
    if isHashTail(name[i + 3 .. name.high]):
      return name[0 .. i - 1]
    i = name.find("___", i + 1)
  result = name

proc isVarDecl(line: string): bool =
  ## Returns true when line is a local declaration statement in a function body.
  ##
  ## - two or more identifier tokens precede the declared name
  ## - termination is `=`, `;`, `{`, or `[[`, an open paren fails it
  ## - statement keywords never declare
  const stmtKeywords = ["return", "break", "continue", "case", "default",
                        "else", "goto", "throw", "delete", "using"]
  let t = line.strip()
  if t.len == 0 or t[0] notin identStart:
    return false
  for kw in stmtKeywords:
    if t.startsWith(kw) and (t.len == kw.len or t[kw.len] notin identChars):
      return false
  var
    i = 0
    tokens = 0
  while i < t.len:
    if t[i] in identStart:
      var j = i
      while j < t.len and t[j] in identChars:
        inc j
      inc tokens
      i = j
    elif t[i] in {' ', '\t'}:
      inc i
    elif t[i] in {'*', '&'}:
      inc i
    elif t[i] == '(':
      return false
    elif t[i] == '[':
      return tokens >= 2 and i + 1 < t.len and t[i + 1] == '['
    elif t[i] in {'=', ';', '{'}:
      return tokens >= 2
    else:
      return false
  result = false

proc countCalls(code, symbol: string): int =
  ## Occurrences of symbol at call position in comment-blanked code text.
  ## Word-bounded on the left, an open paren sits on the symbol's right.
  var p = code.find(symbol)
  while p >= 0:
    let bounded = p == 0 or code[p - 1] notin identChars
    let after = if p + symbol.len < code.len: code[p + symbol.len] else: ' '
    if bounded and after == '(':
      inc result
    p = code.find(symbol, p + 1)

proc scan(msl: string): Scan =
  result.stats.bytes = msl.len
  result.endsWithNewline = msl.endsWith("\n")
  let ls = msl.splitLines()
  result.rawLines = ls
  result.stats.lines = if result.endsWithNewline: ls.len - 1 else: ls.len
  result.codeLines = newSeq[string](ls.len)
  var inBlock = false
  for k in 0 .. ls.high:
    result.codeLines[k] = blankComments(ls[k], inBlock)
  var i = 0
  while i < ls.len:
    let t = ls[i].strip()
    if t.len == 0 or t.startsWith("//") or t.startsWith("#") or
        t.startsWith("using ") or t.startsWith("/*"):
      inc i
      continue
    if t.startsWith("struct") and '{' in t:
      inc result.stats.types
      let head = ls[i].strip()
      let start = i
      var depth = braceDelta(ls[i])
      inc i
      while depth > 0 and i < ls.len:
        depth += braceDelta(ls[i])
        inc i
      var oi: OwnerInfo
      oi.symbol = head["struct ".len .. head.find('{') - 1].strip()
      oi.origin = oi.symbol
      oi.kind = okStruct
      oi.spans.add (start, i - 1)
      result.owners.add oi
      continue
    if '(' in t and t[0] in identStart:
      # function header, `;` terminator marks a prototype, `{` a definition
      var j = i
      var term = ' '
      while j < ls.len:
        let (c, _, _) = scanHeader(ls[j])
        term = c
        if c != ' ':
          break
        inc j
      if term == ';':
        result.protoSpans.add (i, j)
        result.pendingProtos.add PendingProto(symbol: headerName(ls[i]),
                                              span: (i, j))
        i = j + 1
        continue
      if term == '{':
        var fi: FuncInfo
        let hdr = ls[i]
        fi.isInline = hdr.strip().startsWith("inline")
        fi.name = headerName(hdr)
        fi.base = unMangled(fi.name)
        fi.headerLine = i
        var depth = 1
        var k = j + 1
        while k < ls.len and depth > 0:
          depth += braceDelta(ls[k])
          if depth > 0:
            inc fi.loc
            if isVarDecl(ls[k]):
              inc fi.vars
          inc k
        var oi: OwnerInfo
        oi.symbol = fi.name
        oi.origin = fi.base
        oi.kind = okFunction
        # header line through closing brace line
        oi.spans.add (i, k - 1)
        var kept: seq[PendingProto]
        for pp in result.pendingProtos:
          if pp.symbol == fi.name:
            oi.spans.add pp.span
          else:
            kept.add pp
        result.pendingProtos = kept
        result.owners.add oi
        result.funcs.add fi
        inc result.stats.funcs
        if fi.isInline:
          inc result.stats.inlineFuncs
        else:
          inc result.stats.nonInlineFuncs
        inc result.stats.vars, fi.vars
        i = k
        continue
      # no terminator found, malformed emission, step past the opening line
      inc i
      continue
    if isVarDecl(ls[i]):
      var oi: OwnerInfo
      let head = ls[i].strip()
      var e = 0
      while e < head.len and head[e] in identChars:
        inc e
      oi.symbol = head[0 .. e - 1]
      oi.origin = oi.symbol
      oi.kind = okGlobal
      oi.spans.add (i, i)
      result.owners.add oi
      inc i
      continue
    inc i

  # static call-site counts over comment-blanked code, own header line,
  # prototype lines, and struct lines excluded
  var inStructLine = newSeq[bool](ls.len)
  var isProtoLine = newSeq[bool](ls.len)
  for o in result.owners:
    if o.kind == okStruct:
      for span in o.spans:
        for k in span.a .. span.b:
          inStructLine[k] = true
  for span in result.protoSpans:
    for k in span.a .. span.b:
      isProtoLine[k] = true
  for f in result.funcs.mitems:
    var calls = 0
    if f.name.len > 0:
      for k in 0 .. ls.high:
        if k == f.headerLine or inStructLine[k] or isProtoLine[k]:
          continue
        calls += countCalls(result.codeLines[k], f.name)
    f.calls = calls
    inc result.stats.calls, calls

proc addTo(b: var CgsLocBuckets, loc: int) =
  ## Bucket edges are fixed constants for cross-commit comparability, top
  ## edges catch macro/template inlining explosions.
  if loc <= 5:
    inc b.le5
  elif loc <= 30:
    inc b.le30
  elif loc <= 70:
    inc b.le70
  elif loc <= 150:
    inc b.le150
  elif loc <= 300:
    inc b.le300
  elif loc <= 700:
    inc b.le700
  elif loc <= 1400:
    inc b.le1400
  else:
    inc b.gt1400

proc lineBytes(rawLines: seq[string], k: int, endsWithNewline: bool): int =
  ## Byte weight of line k, its line break included, the file's last break
  ## charged to the last line.
  result = rawLines[k].len
  if k < rawLines.high or endsWithNewline:
    inc result

proc analyze*(msl: string): CgsStats =
  ## Size metrics for one MSL source string, per the module counting rules.
  scan(msl).stats

proc locBuckets*(msl: string): CgsLocBuckets =
  ## Function body lines of one MSL source bucketed at the fixed
  ## 5/30/70/150/300/700/1400 upper bounds.
  for f in scan(msl).funcs:
    result.addTo(f.loc)

proc overloadBuckets*(msl: string): CgsOverloadBuckets =
  ## Returns the overload families of one MSL source, bucketed like locBuckets.
  ## groups buckets each family's summed member LOC.
  ## members buckets the individual functions.
  var fams: Table[string, seq[int]]
  var order: seq[string]
  for f in scan(msl).funcs:
    if not fams.hasKey(f.base):
      fams[f.base] = @[]
      order.add f.base
    fams[f.base].add f.loc
  result.families = order.len
  for base in order:
    var total = 0
    for loc in fams[base]:
      total += loc
      result.members.addTo(loc)
    result.groups.addTo(total)

proc attribution*(msl: string): CgsAttribution =
  ## Returns per-owner line attribution of one MSL source.
  ##
  ## Contract:
  ## - every line is attributed to exactly one owner, shares sum to 100%
  ## - a function owner owns its header line, its prototypes, and its body
  ## - calls count per mangled symbol, address-taken symbols stay invisible,
  ##   recursion counts, macro-generated duplicate call names double-count
  let s = scan(msl)
  let n = s.stats.lines
  result.totalLines = n
  result.totalBytes = s.stats.bytes
  var ownerOfLine = newSeq[int](n)
  for k in 0 .. n - 1:
    ownerOfLine[k] = -1
  for oi in 0 .. s.owners.high:
    for span in s.owners[oi].spans:
      for k in span.a .. min(span.b, n - 1):
        ownerOfLine[k] = oi
  var variants: Table[string, int]
  for f in s.funcs:
    variants[f.base] = variants.getOrDefault(f.base, 0) + 1
  var callsOf: Table[string, int]
  for f in s.funcs:
    callsOf[f.name] = f.calls
  var owners: seq[CgsOwner]
  for oi in 0 .. s.owners.high:
    var o: CgsOwner
    o.symbol = s.owners[oi].symbol
    o.origin = s.owners[oi].origin
    o.kind = s.owners[oi].kind
    for span in s.owners[oi].spans:
      for k in span.a .. min(span.b, n - 1):
        inc o.loc
        inc o.bytes, lineBytes(s.rawLines, k, s.endsWithNewline)
    if o.kind == okFunction:
      o.calls = callsOf.getOrDefault(o.symbol, 0)
      o.variants = variants.getOrDefault(o.origin, 1)
    owners.add o
  # unattributed lines, one preamble pseudo-owner
  var pre = -1
  for k in 0 .. n - 1:
    if ownerOfLine[k] < 0:
      if pre < 0:
        pre = owners.len
        owners.add CgsOwner(symbol: "preamble/other", kind: okPreamble)
      inc owners[pre].loc
      inc owners[pre].bytes, lineBytes(s.rawLines, k, s.endsWithNewline)
  owners.sort(proc(x, y: CgsOwner): int = cmp(y.loc, x.loc))
  result.owners = owners
  # overload group summary over Nim origin names
  result.families = variants.len
  for base, count in variants.pairs:
    if count > result.largestFamily:
      result.largestFamily = count
      result.largestFamilyName = base
    if count == 1:
      inc result.singleVariantFamilies
  # minimal top set covering 80% of the lines
  var cum = 0
  for o in result.owners:
    if o.kind == okPreamble:
      continue
    cum += o.loc
    result.pareto.add o.origin
    if n > 0 and cum * 100 >= 80 * n:
      break
  result.paretoLines = cum
  if n > 0:
    result.paretoPct = cum.float * 100.0 / n.float

proc ctr(s: string, w: int): string =
  ## Center s in a field of width w, bencher table-header style.
  let pad = w - s.len
  if pad <= 0:
    return s
  result = " ".repeat(pad div 2) & s & " ".repeat(pad - pad div 2)

proc truncMid(s: string, w: int): string =
  ## s clipped in the middle to w characters, an ellipsis marks the cut,
  ## both ends of a mangled name stay visible.
  if s.len <= w:
    return s
  let head = (w - 3) div 2
  let tail = w - 3 - head
  result = s[0 .. head - 1] & "..." & s[s.len - tail .. ^1]

proc cgsReport*(receipts: openArray[CgsReceipt]) =
  ## Prints the bencher-style codegen-size report for one runner.
  ## Adds totals, the bucket tables, and per-kernel attribution.
  const nameW = 34
  const colSep = "|"
  const lineSep = "|" & "-".repeat(nameW) & "|" & "-".repeat(8) & "|" &
    "-".repeat(6) & "|" & "-".repeat(6) & "|" & "-".repeat(6) & "|" &
    "-".repeat(6) & "|" & "-".repeat(6) & "|" & "-".repeat(10) & "|"
  var total: CgsStats
  var funcB: CgsLocBuckets
  var over: CgsOverloadBuckets
  echo "\n"
  echo colSep, ctr("Kernel", nameW), colSep, ctr("Bytes", 8), colSep,
    ctr("LOC", 6), colSep, ctr("Types", 6), colSep, ctr("Vars", 6),
    colSep, ctr("Funcs", 6), colSep, ctr("calls", 6), colSep,
    ctr("inl/ninl", 10), colSep
  echo lineSep
  for (name, msl) in receipts:
    let s = analyze(msl)
    let inl = $s.inlineFuncs & "/" & $s.nonInlineFuncs
    echo colSep, name.alignLeft(nameW), colSep, ($s.bytes).align(8), colSep,
      ($s.lines).align(6), colSep, ($s.types).align(6), colSep,
      ($s.vars).align(6), colSep, ($s.funcs).align(6), colSep,
      ($s.calls).align(6), colSep, inl.align(10), colSep
    total.bytes += s.bytes
    total.lines += s.lines
    total.types += s.types
    total.vars += s.vars
    total.funcs += s.funcs
    total.inlineFuncs += s.inlineFuncs
    total.nonInlineFuncs += s.nonInlineFuncs
    total.calls += s.calls
    for f in scan(msl).funcs:
      funcB.addTo(f.loc)
    let o = overloadBuckets(msl)
    over.families += o.families
    over.groups.le5 += o.groups.le5
    over.groups.le30 += o.groups.le30
    over.groups.le70 += o.groups.le70
    over.groups.le150 += o.groups.le150
    over.groups.le300 += o.groups.le300
    over.groups.le700 += o.groups.le700
    over.groups.le1400 += o.groups.le1400
    over.groups.gt1400 += o.groups.gt1400
    over.members.le5 += o.members.le5
    over.members.le30 += o.members.le30
    over.members.le70 += o.members.le70
    over.members.le150 += o.members.le150
    over.members.le300 += o.members.le300
    over.members.le700 += o.members.le700
    over.members.le1400 += o.members.le1400
    over.members.gt1400 += o.members.gt1400
  let inl = $total.inlineFuncs & "/" & $total.nonInlineFuncs
  echo colSep, "total".alignLeft(nameW), colSep, ($total.bytes).align(8), colSep,
    ($total.lines).align(6), colSep, ($total.types).align(6), colSep,
    ($total.vars).align(6), colSep, ($total.funcs).align(6), colSep,
    ($total.calls).align(6), colSep, inl.align(10), colSep
  echo "\n"
  echo colSep, ctr("Function LOC", 14), colSep, ctr("Funcs", 10), colSep
  echo "|" & "-".repeat(14) & "|" & "-".repeat(10) & "|"
  let funcRows: seq[(string, int)] = @[("<=5", funcB.le5), ("<=30", funcB.le30),
                                       ("<=70", funcB.le70), ("<=150", funcB.le150),
    ("<=300", funcB.le300), ("<=700", funcB.le700), ("<=1400", funcB.le1400),
    (">1400", funcB.gt1400)]
  for (label, n) in funcRows:
    echo colSep, label.align(14), colSep, ($n).align(10), colSep
  echo "\n"
  echo colSep, ctr("Overload LOC", 14), colSep, ctr("Families", 10), colSep,
    ctr("Members", 10), colSep
  echo "|" & "-".repeat(14) & "|" & "-".repeat(10) & "|" & "-".repeat(10) & "|"
  let overRows: seq[(string, int, int)] = @[("<=5", over.groups.le5, over.members.le5),
      ("<=30", over.groups.le30, over.members.le30),
      ("<=70", over.groups.le70, over.members.le70),
      ("<=150", over.groups.le150, over.members.le150),
      ("<=300", over.groups.le300, over.members.le300),
      ("<=700", over.groups.le700, over.members.le700),
      ("<=1400", over.groups.le1400, over.members.le1400),
      (">1400", over.groups.gt1400, over.members.gt1400)]
  for (label, g, m) in overRows:
    echo colSep, label.align(14), colSep, ($g).align(10), colSep,
      ($m).align(10), colSep
  echo colSep, ctr("families", 14), colSep, ($over.families).align(10), colSep,
    ($total.funcs).align(10), colSep
  # per-kernel line attribution
  const ownW = 32
  const orgW = 20
  const attSep = "|" & "-".repeat(ownW) & "|" & "-".repeat(orgW) & "|" &
    "-".repeat(6) & "|" & "-".repeat(6) & "|" & "-".repeat(6) & "|" &
    "-".repeat(6) & "|" & "-".repeat(8) & "|" & "-".repeat(7) & "|" &
    "-".repeat(7) & "|" & "-".repeat(4) & "|"
  for (name, msl) in receipts:
    let a = attribution(msl)
    var ownerCount = 0
    for o in a.owners:
      if o.kind != okPreamble:
        inc ownerCount
    echo "\nAttribution for ", name, ", ", $a.totalLines, " lines, ",
      $ownerCount, " owners"
    echo colSep, ctr("Owner (mangled)", ownW), colSep, ctr("Nim origin", orgW),
      colSep, ctr("LOC", 6), colSep, ctr("LOC%", 6), colSep, ctr("cum%", 6),
      colSep, ctr("calls", 6), colSep, ctr("Bytes", 8), colSep,
      ctr("B/call", 7), colSep, ctr("L/call", 7), colSep, ctr("ovl", 4), colSep
    echo attSep
    var cum = 0
    for o in a.owners:
      cum += o.loc
      let pct = if a.totalLines > 0: o.loc.float * 100.0 / a.totalLines.float else: 0.0
      let cpct = if a.totalLines > 0: cum.float * 100.0 / a.totalLines.float else: 0.0
      let bcall = if o.calls > 0: ($((o.bytes div o.calls))).align(7) else: "-".align(7)
      let lcall = if o.calls > 0: ($((o.loc div o.calls))).align(7) else: "-".align(7)
      let ovl = if o.variants > 0: ($o.variants).align(4) else: "-".align(4)
      let org = if o.origin.len > 0: truncMid(o.origin, orgW) else: "-"
      echo colSep, truncMid(o.symbol, ownW).alignLeft(ownW), colSep,
        org.alignLeft(orgW), colSep, ($o.loc).align(6), colSep,
        (&"{pct:.1f}").align(6), colSep, (&"{cpct:.1f}").align(6), colSep,
        ($o.calls).align(6), colSep, ($o.bytes).align(8), colSep,
        bcall, colSep, lcall, colSep, ovl, colSep
    let exhausted = a.paretoPct < 80.0 and a.pareto.len == ownerCount
    echo "pareto 80%: ", $a.pareto.len, " of ", $ownerCount, " owners cover ",
      (&"{a.paretoPct:.1f}"), "% of lines", 
      if exhausted: " (all owners, the remainder is preamble)" else: "",
      ": ", a.pareto.join(", ")
    echo "overload groups: ", $a.families, ", largest ", $a.largestFamily,
      " variants (", a.largestFamilyName, "), ", $a.singleVariantFamilies,
      " with a single variant"
  echo "\nattribution caveats:"
  echo "- helpers are attributed per kernel, a cross-kernel dedupe is a compare-level view"
  echo "- a Nim def inlined at N call sites emits N functions, the table reports emitted code"
  echo "- call counts miss address-taken symbols, count recursive calls, and"
  echo "  can double-count macro-generated duplicate call names"
  echo ""

proc cgsCompare*(name: string, before, after: CgsStats) =
  ## Prints the side-by-side delta report of two measurements, delta is
  ## after minus before.
  const lineSep = "|" & "-".repeat(16) & "|" & "-".repeat(10) & "|" &
    "-".repeat(10) & "|" & "-".repeat(10) & "|"
  echo "\n", name
  echo "|", ctr("Metric", 16), "|", ctr("before", 10), "|",
    ctr("after", 10), "|", ctr("delta", 10), "|"
  echo lineSep
  let rows: seq[(string, int, int)] = @[("bytes", before.bytes, after.bytes),
      ("lines", before.lines, after.lines), ("types", before.types, after.types),
      ("vars", before.vars, after.vars), ("funcs", before.funcs, after.funcs),
      ("calls", before.calls, after.calls),
      ("inline", before.inlineFuncs, after.inlineFuncs),
      ("non-inline", before.nonInlineFuncs, after.nonInlineFuncs)]
  for (label, b, a) in rows:
    let d = a - b
    let signed = if d >= 0: "+" & $d else: $d
    echo "|", label.align(16), "|", ($b).align(10), "|", ($a).align(10), "|",
      signed.align(10), "|"
  echo ""
