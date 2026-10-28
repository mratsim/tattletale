# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## HLSL printer: a structured CIRv2 module rendered as HLSL compute source for DXC,
## table-driven over the closed type lattice.
## Two resource-binding flavors share one language body, the binding section differs.
##
## Emission contract:
## - one `{.global.}` kernel per module, emitted as `main`,
##   DXC compiles one entry point per invocation with `-E main`,
##   device functions alongside, forward-declared
##
## fvD3D12 (default, D3D12-native binding):
## - buffer parameters bind StructuredBuffer / RWStructuredBuffer resources,
##   read-only vs read-write, on one global register ordinal in parameter order
##   (`t` vs `u`), the write-scan is conservative: a buffer counts as read-write
##   when the kernel or any device function argument stores through it.
## - scalars and value structs pack into one `struct Push` block behind the buffers:
##   `ConstantBuffer<Push> push : register(bN, space0)` where N is the buffer count,
##   scalar reads go through `push.<dotted path>`
##
## fvVulkan (zero-descriptor, SPIR-V through the engine's address route):
## - every kernel parameter is a member of one `[[vk::push_constant]]` struct,
##   buffer parameters as `uint64_t` device addresses, every member offset
##   locked with `[[vk::offset]]` to the launch-side packing
## - the launch-side packing: width-aligned scalars and value structs, the staged grid member 16-byte aligned
## - a buffer access constructs `vk::BufferPointer<E>` from the address per use,
##   an lvalue for reads and writes, device-function buffer parameters take the address by value
## - buffer elements: 32-bit integers, float, double, a buffer parameter is
##   writable regardless of the write-scan, the BDA route has no read-only
##   binding to violate
##
## Shared:
## - the dispatch grid size has no HLSL semantic: a `threadgroups_per_grid` read refuses,
##   the HLSL lowering pass materializes the read into the trailing `crucible_numworkgroups` parameter,
##   staged at launch as a descriptor-route uniform on fvD3D12, as a push block member on fvVulkan
##
## Addressing, memory classes, jumps:
## - HLSL has no pointer values: pointer locals, pointer-holding struct locals,
##   `var` parameter derefs all dissolve in the HLSL lowering pass,
##   anything pointer shaped reaching the printer refuses
## - `{.const_mem.}` locals relocate into `CirModule.globals` through the HLSL lowering pass,
##   file-scope `static const` output, a smem, device-space or surviving
##   constant-space local refuses
## - jmp to the loop exit label is `break`, to the continue label `continue`,
##   any other lab/jmp refuses
## - object constructions emit flat brace lists at any depth, `{ 0, 1, 2 }`,
##   the form DXC accepts over uniformly scalar members, a constructor assignment
##   lowers to member-wise leaves, DXC rejects compound literals on assignment.

import std/[sets, strutils, tables]

import ../ir/cirv2_lang
import ../ir/cirv2_pretty
import ../frontend/nim_to_crucible
import ../passes/pass_helpers
import ./targets

# ═══════════════════════════════════════════════════════════════════════
# Per-proc emission state
# ═══════════════════════════════════════════════════════════════════════

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    isKernel: bool
    ## Scalar kernel parameters, their reads go through `push.<name>`.
    pushParams: HashSet[string]
    ## Buffer parameters of the proc in flight, bare use refuses.
    bufferParams: HashSet[string]
    ## Buffer parameters classified read-write by the module write-scan.
    rwBuffers: HashSet[string]
    flavor: HlslFlavor
    ## Buffer parameters of the proc in flight by name, to their `aptr` type:
    ## the fvVulkan access emitter reads the element type per access.
    addressParams: Table[string, CirType]
    ## Coordinate semantics the kernel body reads, main takes only those.
    coordReads: set[CoordBuiltinKind]
    workgroup: array[3, int]
    procs: Table[string, CirProc]
    ## Module procs by name, by-reference call arguments read callee
    ## parameter types here.

proc failing(ctx: PrintCtx): bool =
  ctx.err.len > 0

proc fail(ctx: var PrintCtx, detail: string) =
  ## Records the first refusal, naming the proc.
  if ctx.err.len == 0:
    ctx.err = "cirv2 printer hlsl: at " & ctx.fnName & ": " & detail

proc toPlainSym(name: string, typ: CirType): CirExpr =
  ## Plain symbol node without a builtin kind, the rewrite target of every dissolved read.
  result = CirExpr(kind: eSym, name: name, typ: typ, builtin: false,
    coordBuiltin: cbkNone, synchroBuiltin: sbkNone,
    reductionBuiltin: rbkNone, atomicsBuiltin: abkNone)

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc spellHlslScalar(ctx: var PrintCtx, t: CirType): string =
  ## HLSL name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid:
    result = "void"
  of ctBool:
    result = "bool"
  of ctInt:
    case t.bits
    of 32:
      result = "int"
    else:
      ctx.fail("int" & $t.bits & " has no HLSL type name: the 8/16-bit forms" &
        " are parse errors and int64_t is a parse error in the HLSL frontend," &
        " 64-bit integers are rejected")
  of ctUint:
    case t.bits
    of 32:
      result = "uint"
    else:
      ctx.fail("uint" & $t.bits & " has no HLSL type name: the 8/16-bit forms" &
        " are parse errors and uint64_t pulls the SPIR-V Int64 capability," &
        " 64-bit integers are rejected")
  of ctFloat:
    case t.bits
    of 32:
      result = "float"
    of 64:
      result = "double"
    else:
      ctx.fail("float16 has no HLSL type name, float16 types are outside the HLSL subset")
  of ctVector:
    let elem = spellHlslScalar(ctx, t.vElem)
    let base = case elem
      of "float": "float"
      of "int": "int"
      of "uint": "uint"
      of "bool": "bool"
      else: ""
    if base.len > 0 and t.vN in 2 .. 4:
      result = base & $t.vN
    else:
      ctx.fail("vector of " & (if elem.len > 0: elem else: $t.vElem.kind) &
        "x" & $t.vN & " has no HLSL type name, only 2..4-wide float/int/uint/bool vectors spell")
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc:
    result = ""

proc spellHlslTypeBare(ctx: var PrintCtx, t: CirType): string =
  ## HLSL name of `t` with no identifier, the form of casts and constructors.
  ## Arrays have no bare form, `T name[N]` composes at the declaration.
  case t.kind
  of ctArray:
    ctx.fail("array type has no bare HLSL spelling, arrays compose at their declaration")
  else:
    result = spellHlslScalar(ctx, t)

proc spellHlslDecl(ctx: var PrintCtx, t: CirType, ident: string): string =
  ## HLSL declaration of `t` for `ident`, array dimensions composed into the name.
  if t.kind == ctArray:
    result = spellHlslScalar(ctx, t.aElem) & ' ' & ident & "[" & $t.aN & "]"
  else:
    result = spellHlslScalar(ctx, t) & ' ' & ident

proc roundUpMultiple(v, multiple: int): int =
  ## Smallest multiple of `multiple` that is >= `v`
  if multiple <= 1:
    return v
  result = (v + multiple - 1) div multiple * multiple

proc spellBpName(t: CirType): string =
  ## Typedef name of the `vk::BufferPointer` alias for one buffer element type
  case t.kind
  of ctInt: result = "BpI32"
  of ctUint: result = "BpU32"
  of ctFloat: result = (if t.bits == 64: "BpF64" else: "BpF32")
  else: result = ""

proc bpElementStride(ctx: var PrintCtx, elem: CirType): int =
  ## Byte stride of one buffer element on the fvVulkan route, its storage width.
  ## The address arithmetic indexes buffers by it.
  case elem.kind
  of ctInt, ctUint:
    if elem.bits != 32:
      ctx.fail("buffer element of " & $elem.bits & "-bit integer has no fvVulkan" &
        " binding, the element envelope is 32-bit integers, float and double")
    result = 4
  of ctFloat:
    if elem.bits != 32 and elem.bits != 64:
      ctx.fail("buffer element of float" & $elem.bits & " has no fvVulkan" &
        " binding, the element envelope is 32-bit integers, float and double")
    result = elem.bits div 8
  else:
    ctx.fail("buffer element of " & $elem.kind & " type has no fvVulkan binding," &
      " the element envelope is 32-bit integers, float and double")
    result = 0

proc pushMemberExtent(ctx: var PrintCtx, t: CirType, label: string): tuple[size, align: int] =
  ## Packed size and alignment of one fvVulkan push-block value member.
  ## Member offsets lock to the launch-side packing: scalars and vectors are
  ## width-aligned, struct members stay flat over scalars, the alignment is
  ## the max field width, the size rounds to it.
  case t.kind
  of ctInt, ctUint, ctFloat:
    let w = t.bits div 8
    result = (w, w)
  of ctBool:
    ctx.fail("bool push member " & label & ", the fvVulkan push block has no" &
      " bool member")
  of ctVector:
    let w = t.vElem.bits div 8
    result = (w * t.vN, w)
  of ctStruct:
    result.align = 1
    var size = 0
    for f in t.fields:
      case f.typ.kind
      of ctInt, ctUint, ctFloat:
        let w = f.typ.bits div 8
        size = roundUpMultiple(size, w) + w
        result.align = max(result.align, w)
      else:
        ctx.fail("field " & f.name & " of push member " & label & " is a " &
          $f.typ.kind & ", the fvVulkan push block packs struct members flat" &
          " over 32/64-bit scalars: the compiler's internal struct layout" &
          " (16-byte struct alignment, the vector register-straddle rule)" &
          " diverges from the launch packing for anything wider")
    size = roundUpMultiple(size, result.align)
    result.size = size
  else:
    ctx.fail("push member " & label & " of " & $t.kind & " type has no fvVulkan binding")

# ═══════════════════════════════════════════════════════════════════════
# Write-scan: read-write classification and the call graph
# ═══════════════════════════════════════════════════════════════════════

type
  CallGraph = object
    procs: Table[string, CirProc]
    ## Module procs by name.
    edges: seq[tuple[caller, arg, callee, param: string]]
    ## One argument flow: `caller` passes the argument rooted at parameter
    ## `arg` to the `param` parameter of `callee`.
    stored: HashSet[string]
    ## `fn/param` keys whose parameter the proc stores through directly.
    reached: HashSet[string]
    ## `fn/param` keys reachable from `stored` over the argument-flow edges.

proc exprRootSym(e: CirExpr): CirExpr =
  ## Symbol at the root of a dotted or indexed chain, nil otherwise.
  result = e
  while result != nil and result.kind in {eDot, ePat, eAt}:
    result = case result.kind
      of eDot: result.dotBase
      of ePat: result.patBase
      else: result.atBase
  if result != nil and result.kind != eSym:
    result = nil

proc directCalls(ss: seq[CirStmt], outp: var seq[CirExpr]) =
  ## Call expressions of one statement list, a call binds to its destination:
  ## the sites are statement calls plus the right-hand sides of assignments,
  ## declarations and returns.
  for s in ss:
    case s.kind
    of sCall:
      outp.add CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
    of sAsgn:
      if s.rhs != nil and s.rhs.kind == eCall:
        outp.add s.rhs
    of sVar, sConst:
      if s.init != nil and s.init.kind == eCall:
        outp.add s.init
    of sRet:
      if s.retVal != nil and s.retVal.kind == eCall:
        outp.add s.retVal
    of sIte:
      directCalls(s.thenBody, outp)
      directCalls(s.elseBody, outp)
    of sLoop:
      directCalls(s.loopBody, outp)
    of sScope:
      directCalls(s.scopeBody, outp)
    of sJmp, sLab, sAssume, sDiscard, sCase, sAsm:
      discard

proc scanStoredStmts(graph: var CallGraph, fn: CirProc, ss: seq[CirStmt]) =
  ## Records the parameters `fn` stores through: an assignment whose target
  ## roots at a parameter symbol, anywhere in the statement tree.
  for s in ss:
    case s.kind
    of sAsgn:
      let root = exprRootSym(s.lhs)
      if root != nil:
        for p in fn.params:
          if p.name == root.name:
            graph.stored.incl(fn.name & "/" & p.name)
    of sIte:
      graph.scanStoredStmts(fn, s.thenBody)
      graph.scanStoredStmts(fn, s.elseBody)
    of sLoop:
      graph.scanStoredStmts(fn, s.loopBody)
    of sScope:
      graph.scanStoredStmts(fn, s.scopeBody)
    of sCall:
      # a store can hide in a call-bound destination expression
      let root = exprRootSym(s.callCallee)
      if root != nil:
        for p in fn.params:
          if p.name == root.name:
            graph.stored.incl(fn.name & "/" & p.name)
      for a in s.callArgs:
        let ar = exprRootSym(a)
        if ar != nil:
          for p in fn.params:
            if p.name == ar.name:
              graph.stored.incl(fn.name & "/" & p.name)
    of sVar, sConst, sJmp, sLab, sAssume, sRet, sDiscard, sCase, sAsm:
      discard

proc scanCallStmts(graph: var CallGraph, fn: CirProc, ss: seq[CirStmt]) =
  ## Records one edge per call argument rooted at a parameter symbol:
  ## `caller/arg -> callee/paramI`.
  for s in ss:
    case s.kind
    of sCall:
      let calleeName = if s.callCallee.kind == eSym: s.callCallee.name else: ""
      if calleeName.len > 0 and graph.procs.hasKey(calleeName):
        let callee = graph.procs[calleeName]
        for i, a in s.callArgs:
          let root = exprRootSym(a)
          if root == nil:
            continue
          for p in fn.params:
            if p.name == root.name:
              if i < callee.params.len:
                graph.edges.add (fn.name, p.name, calleeName, callee.params[i].name)
    of sAsgn:
      if s.rhs != nil and s.rhs.kind == eCall:
        graph.scanCallStmts(fn, @[CirStmt(kind: sCall, callCallee: s.rhs.callee,
          callArgs: s.rhs.args)])
    of sIte:
      graph.scanCallStmts(fn, s.thenBody)
      graph.scanCallStmts(fn, s.elseBody)
    of sLoop:
      graph.scanCallStmts(fn, s.loopBody)
    of sScope:
      graph.scanCallStmts(fn, s.scopeBody)
    of sVar, sConst, sJmp, sLab, sAssume, sRet, sDiscard, sCase, sAsm:
      discard

proc classifyRwBuffers(graph: var CallGraph) =
  ## Marks every buffer parameter stored through, directly or transitively
  ## over the argument-flow edges: a read-write fixpoint over the call graph.
  graph.reached = graph.stored
  var grew = true
  while grew:
    grew = false
    for e in graph.edges:
      let src = e.caller & "/" & e.arg
      let dst = e.callee & "/" & e.param
      if src in graph.reached and dst notin graph.reached:
        graph.reached.incl(dst)
        grew = true

proc checkDeviceFnRecursion(ctx: var PrintCtx, graph: CallGraph) =
  ## Refuses recursion over the device-function call graph before emission:
  ## a recursive call would not terminate, and DXC accepts recursive source
  ## with exit 0 while miscompiling it.
  var state = initTable[string, int]() # fn name: 1 = on the visit stack, 2 = fully visited

  proc visit(ctx: var PrintCtx, graph: CallGraph, state: var Table[string, int], calleesOf: Table[string, seq[string]], fn: string) =
    state[fn] = 1
    for callee in calleesOf.getOrDefault(fn, @[]):
      case state.getOrDefault(callee, 0)
      of 1:
        ctx.fail("recursive device function `" & callee &
          "` would not terminate, recursion is rejected")
      of 0:
        visit(ctx, graph, state, calleesOf, callee)
      else:
        discard
    state[fn] = 2

  var calleesOf: Table[string, seq[string]]
  for fn in graph.procs.values:
    if fn.isKernel:
      continue
    var calls: seq[CirExpr]
    directCalls(fn.body, calls)
    for c in calls:
      if c.callee.kind == eSym and graph.procs.hasKey(c.callee.name) and
          not graph.procs[c.callee.name].isKernel:
        calleesOf.mgetOrPut(fn.name, @[]).add(c.callee.name)
  for fn in graph.procs.values:
    if not fn.isKernel and state.getOrDefault(fn.name, 0) == 0:
      visit(ctx, graph, state, calleesOf, fn.name)
    if ctx.failing():
      return

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

## Synthesized main input variable per coordinate builtin kind, "" for the kinds
## without one: threads_per_threadgroup reads the workgroup literal,
## threadgroups_per_grid materializes in the lowering pass.
const CoordInputNames: array[CoordBuiltinKind, string] = [
  "", "gid", "groupId", "groupThreadId", "", "", "gidx",
]

proc coordInputName(kind: CoordBuiltinKind): string =
  ## Synthesized main input variable of a coordinate builtin kind.
  CoordInputNames[kind]

proc spellCoordWhole(ctx: var PrintCtx, kind: CoordBuiltinKind): string =
  ## Whole-vector read of a coordinate builtin, cast to the canonical int
  ## tuple the CIRv2 vocabulary reads.
  ctx.coordReads.incl(kind)
  case kind
  of cbkThreadsPerThreadgroup:
    result = "int3(" & $ctx.workgroup[0] & ", " & $ctx.workgroup[1] & ", " &
      $ctx.workgroup[2] & ")"
  of cbkThreadIndexInThreadgroup:
    result = "int(" & coordInputName(kind) & ")"
  else:
    let n = coordInputName(kind)
    result = "int3(int(" & n & ".x), int(" & n & ".y), int(" & n & ".z))"

proc spellCoordComponent(ctx: var PrintCtx, kind: CoordBuiltinKind, field: string): string =
  ## One coordinate-builtin component, cast to the canonical int the CIRv2 vocabulary reads.
  ctx.coordReads.incl(kind)
  case kind
  of cbkThreadsPerThreadgroup:
    let i = case field
      of "x": 0
      of "y": 1
      of "z": 2
      else: -1
    if i < 0:
      ctx.fail("no component " & field & " on threads_per_threadgroup")
    result = $ctx.workgroup[i]
  of cbkThreadIndexInThreadgroup:
    result = "int(" & coordInputName(kind) & ")"
  else:
    result = "int(" & coordInputName(kind) & "." & field & ")"

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(ctx: var PrintCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc spellHlslLit(e: CirExpr): string =
  ## HLSL token of a literal, typed by its literal type: without the suffix
  ## the HLSL frontend changes width or signedness.
  case e.kind
  of eIntLit:
    case e.litType.kind
    of ctUint:
      result = $e.value & "u"
    else:
      result = $e.value
  of eFloatLit:
    case e.fLitType.bits
    of 64:
      result = $e.fValue & "l"
    else:
      result = $e.fValue & "f"
  of eTrue: result = "true"
  of eFalse: result = "false"
  else: result = "<hlsl: not a literal>"

proc spellVectorComponent(name: string): string =
  ## HLSL component letter of a CIRv2 vector component name (`F0`..`F3`).
  case name
  of "F0": "x"
  of "F1": "y"
  of "F2": "z"
  of "F3": "w"
  else: ""

proc printBraceList(ctx: var PrintCtx, t: CirType, v: CirExpr): string
  ## Forward declaration, nested constructions flatten recursively.

proc braceLeaves(ctx: var PrintCtx, t: CirType, v: CirExpr, outp: var seq[string]) =
  ## Appends the flat leaf values of one construction over struct type `t`,
  ## fields in declaration order, nested constructions flatten recursively.
  ## A zero-field struct contributes one `0` slot for its padding member.
  if t.fields.len == 0:
    outp.add "0"
    return
  for f in t.fields:
    var val: CirExpr = nil
    for kvv in v.kvFields:
      if kvv.name == f.name:
        val = kvv.value
    if val == nil:
      ctx.fail("field " & f.name & " of " & t.name & " is not bound by the construction")
      return
    if val.kind == eOconstr and f.typ.kind == ctStruct:
      ctx.braceLeaves(f.typ, val, outp)
    else:
      outp.add ctx.printExpr(val)

proc printBraceList(ctx: var PrintCtx, t: CirType, v: CirExpr): string =
  ## Flat brace list of one object construction: `{ 0, 1, 2 }` at any depth.
  var leaves: seq[string]
  ctx.braceLeaves(t, v, leaves)
  if ctx.failing():
    return ""
  result = "{ "
  result.add leaves.join(", ")
  result.add " }"

proc spellCallArg(ctx: var PrintCtx, callee: CirExpr, i: int, a: CirExpr): string =
  ## One call argument: an address-of over a variable argument to a by-reference parameter
  ## emits the variable itself, the `inout` binding takes lvalues directly.
  if a.kind == eAddr and a.addrOperand.kind == eSym and
      callee.kind == eSym and ctx.procs.hasKey(callee.name):
    let fn = ctx.procs[callee.name]
    if i < fn.params.len and fn.params[i].typ.kind == ctPtr and
        fn.params[i].typ.space == caThread:
      return a.addrOperand.name
  return ctx.printExpr(a)

proc printExpr(ctx: var PrintCtx, e: CirExpr): string =
  if ctx.failing():
    return ""
  case e.kind
  of eNil:
    ctx.fail("nil literal is outside the HLSL subset")
    return ""
  of eTrue, eFalse, eIntLit, eFloatLit:
    return spellHlslLit(e)
  of eSym:
    if e.coordBuiltin != cbkNone:
      if not ctx.isKernel:
        ctx.fail("coordinate builtin in a device function, coordinate" &
          " semantics bind the kernel entry point only")
        return ""
      if e.coordBuiltin == cbkThreadgroupsPerGrid:
        ctx.fail("threadgroups_per_grid has no HLSL semantic, the HLSL" &
          " lowering pass materializes it as the " & "crucible_numworkgroups" &
          " parameter")
        return ""
      return spellCoordWhole(ctx, e.coordBuiltin)
    if ctx.pushParams.contains(e.name):
      return "push." & e.name
    if ctx.bufferParams.contains(e.name):
      if ctx.flavor == fvVulkan:
        # zero-descriptor route: the parameter IS its u64 device address
        return (if ctx.isKernel: "push." & e.name else: e.name)
      ctx.fail("bare pointer value use of buffer parameter " & e.name)
      return ""
    return e.name
  of eAt, ePat:
    let base = (if e.kind == eAt: e.atBase else: e.patBase)
    let idx = ctx.printExpr(if e.kind == eAt: e.atIndex else: e.patIndex)
    if ctx.flavor == fvVulkan and base.kind == eSym and
        ctx.addressParams.hasKey(base.name):
      let elem = ctx.addressParams[base.name].pElem
      let stride = ctx.bpElementStride(elem)
      if ctx.failing():
        return ""
      return spellBpName(elem) & "(" & ctx.printExpr(base) &
        " + (uint64_t)(" & idx & ") * " & $stride & ").Get()"
    if base.kind == eSym and ctx.bufferParams.contains(base.name):
      return base.name & "[" & idx & "]"
    let bt = getExprType(base)
    if bt != nil and bt.kind == ctArray:
      return ctx.printExpr(base) & "[" & idx & "]"
    ctx.fail("index of a " & (if bt == nil: "untyped" else: $bt.kind) &
      " base is outside the HLSL subset")
    return ""
  of eDeref:
    ctx.fail("bare dereference has no emission, HLSL has no pointer values")
    return ""
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      if not ctx.isKernel:
        ctx.fail("coordinate builtin in a device function, coordinate" &
          " semantics bind the kernel entry point only")
        return ""
      if e.dotBase.coordBuiltin == cbkThreadgroupsPerGrid:
        ctx.fail("threadgroups_per_grid has no HLSL semantic, the HLSL" &
          " lowering pass materializes it as the " & "crucible_numworkgroups" &
          " parameter")
        return ""
      return spellCoordComponent(ctx, e.dotBase.coordBuiltin, e.fieldName)
    let bt = getExprType(e.dotBase)
    if bt != nil and bt.kind == ctVector:
      let comp = spellVectorComponent(e.fieldName)
      if comp.len == 0:
        ctx.fail("no vector component " & e.fieldName)
        return ""
      return ctx.printExpr(e.dotBase) & "." & comp
    return ctx.printExpr(e.dotBase) & "." & e.fieldName
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    let op = case e.kind
      of eAdd: "+"
      of eSub: "-"
      of eMul: "*"
      of eDiv: "/"
      of eMod: "%"
      of eShl: "<<"
      of eShr: ">>"
      of eBitAnd: "&"
      of eBitOr: "|"
      else: "^"
    return "(" & ctx.printExpr(e.left) & ' ' & op & ' ' & ctx.printExpr(e.right) & ")"
  of eEq, eNeq, eLt, eLe:
    let op = case e.kind
      of eEq: "=="
      of eNeq: "!="
      of eLt: "<"
      else: "<="
    return "(" & ctx.printExpr(e.a) & ' ' & op & ' ' & ctx.printExpr(e.b) & ")"
  of eNot:
    return "(!" & ctx.printExpr(e.notOperand) & ")"
  of eAnd, eOr:
    let op = if e.kind == eAnd: "&&" else: "||"
    return "(" & ctx.printExpr(e.andLeft) & ' ' & op & ' ' & ctx.printExpr(e.andRight) & ")"
  of eBitNot:
    return "(~" & ctx.printExpr(e.unOperand) & ")"
  of eNeg:
    return "(-" & ctx.printExpr(e.unOperand) & ")"
  of eCast:
    let t = e.castType
    let fn = if t.kind == ctInt and t.bits == 32: "asint"
      elif t.kind == ctUint and t.bits == 32: "asuint"
      elif t.kind == ctFloat and t.bits == 32: "asfloat"
      else: ""
    if fn.len == 0:
      ctx.fail("bit cast to " & spellHlslTypeBare(ctx, t) &
        " has no HLSL emission, only the 32-bit asint/asuint/asfloat reinterprets")
      return ""
    return fn & "(" & ctx.printExpr(e.castOperand) & ")"
  of eConv:
    return spellHlslTypeBare(ctx, e.castType) & "(" & ctx.printExpr(e.castOperand) & ")"
  of eSizeof:
    ctx.fail("sizeof has no HLSL emission, HLSL struct layout differs from the" &
      " buffer layout the size would name")
    return ""
  of eAddr:
    ctx.fail("address-of has no HLSL form, buffers are index-only resources" &
      " and locals dissolve to their pointee")
    return ""
  of eOconstr:
    if e.ocType.kind == ctVector:
      # vectors construct by call, valid in declaration initializers and assignments alike,
      # where struct brace lists only initialize
      var parts: seq[string]
      for kvv in e.kvFields:
        parts.add ctx.printExpr(kvv.value)
      return spellHlslScalar(ctx, e.ocType) & "(" & parts.join(", ") & ")"
    return ctx.printBraceList(e.ocType, e)
  of eAconstr:
    var parts: seq[string]
    for el in e.elems:
      parts.add ctx.printExpr(el)
    return "{ " & parts.join(", ") & " }"
  of eCall:
    let calleeName =
      if e.callee.kind == eSym: e.callee.name
      else: ctx.printExpr(e.callee)
    var args: seq[string]
    for i, a in e.args:
      if ctx.flavor == fvD3D12 and a.kind == eSym and
          ctx.bufferParams.contains(a.name):
        # a buffer resource passes by name, its identity flows from the call site
        args.add a.name
      else:
        args.add ctx.spellCallArg(e.callee, i, a)
    if e.callee.kind == eSym:
      case e.callee.synchroBuiltin
      of sbkThreadgroupBarrier, sbkThreadgroupBarrierDevice:
        ctx.fail("barrier is a statement, a value-position barrier is outside the HLSL subset")
        return ""
      else:
        discard
      if e.callee.reductionBuiltin != rbkNone:
        ctx.fail("subgroup reductions have no emission, wave intrinsics are outside this backend")
        return ""
      if e.callee.atomicsBuiltin != abkNone:
        ctx.fail("atomic builtins have no emission in this backend")
        return ""
    return calleeName & "(" & args.join(", ") & ")"

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string
  ## Forward declaration, the loop context (break/continue labels) is a parameter.

proc printObjAssign(ctx: var PrintCtx, target: string, t: CirType, v: CirExpr, lines: var seq[string]) =
  ## Appends one `target.leaf = value` line per leaf field of a struct construction,
  ## nested constructions recurse. DXC rejects compound literals on assignment,
  ## member-wise leaves are the form it accepts.
  for f in t.fields:
    var val: CirExpr = nil
    for kvv in v.kvFields:
      if kvv.name == f.name:
        val = kvv.value
    if val == nil:
      ctx.fail("field " & f.name & " of " & t.name & " is not bound by the construction")
      return
    if val.kind == eOconstr and f.typ.kind == ctStruct:
      ctx.printObjAssign(target & "." & f.name, f.typ, val, lines)
    else:
      lines.add target & "." & f.name & " = " & ctx.printExpr(val) & ";"

proc printStmt(ctx: var PrintCtx, s: CirStmt, exitL, contL: string): string =
  ## One structured statement, `exitL`/`contL` the enclosing loop's labels.
  if ctx.failing():
    return ""
  let pad = spaces(ctx.indent * 2)
  case s.kind
  of sVar:
    if s.declType.kind in {ctPtr, ctAptr}:
      ctx.fail("pointer local " & s.name & ", HLSL has no pointer values," &
        " the HLSL lowering pass dissolves pointer locals")
      return ""
    if s.space == caThreadgroup:
      ctx.fail("local " & s.name & " in threadgroup space, HLSL has no" &
        " program-scope shared emission")
      return ""
    if s.space == caConstant:
      ctx.fail("local " & s.name & " in constant space, the HLSL lowering" &
        " pass hoists const_mem locals to program scope")
      return ""
    if s.space == caDevice:
      ctx.fail("local " & s.name & " in device space, HLSL locals are per-thread registers")
      return ""
    var decl = ctx.spellHlslDecl(s.declType, s.name)
    if s.init != nil:
      if s.init.kind == eOconstr and s.declType.kind == ctStruct:
        decl &= " = " & ctx.printBraceList(s.declType, s.init)
      else:
        decl &= " = " & ctx.printExpr(s.init)
    result = pad & decl & ";"
  of sConst:
    if s.init != nil and s.init.kind == eOconstr and s.declType.kind == ctStruct:
      result = pad & "const " & ctx.spellHlslDecl(s.declType, s.name) &
        " = " & ctx.printBraceList(s.declType, s.init) & ";"
    else:
      result = pad & "const " & ctx.spellHlslDecl(s.declType, s.name) &
        " = " & ctx.printExpr(s.init) & ";"
  of sAsgn:
    let root = exprRootSym(s.lhs)
    if root != nil and ctx.bufferParams.contains(root.name):
      if ctx.flavor == fvD3D12 and root.name notin ctx.rwBuffers:
        ctx.fail("assignment through read-only buffer parameter " & root.name)
        return ""
      result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"
      return
    if s.lhs.kind == eSym and ctx.pushParams.contains(s.lhs.name):
      ctx.fail("assignment to push-constant parameter " & s.lhs.name &
        ", push constants are read-only inputs")
      return ""
    if s.rhs != nil and s.rhs.kind == eOconstr and s.rhs.ocType.kind == ctStruct:
      # member-wise leaf assignment, DXC rejects compound literals on assignment
      var lines: seq[string]
      ctx.printObjAssign(ctx.printExpr(s.lhs), s.rhs.ocType, s.rhs, lines)
      if ctx.failing():
        return ""
      result = lines.join("\n")
      return
    if s.rhs != nil and s.rhs.kind == eAconstr:
      ctx.fail("array constructor assignment has no HLSL form")
      return ""
    result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"
  of sIte:
    let cond = ctx.printExpr(s.cond)
    ctx.indent.inc
    let thenTxt = ctx.printStmts(s.thenBody, exitL, contL)
    ctx.indent.dec
    result = pad & "if (" & cond & ") {\n" & thenTxt & '\n' & pad & '}'
    if s.elseBody.len == 1 and s.elseBody[0].kind == sIte:
      # an else side of one ite is the elif chain, printed as `else if`
      result &= " else " & ctx.printStmt(s.elseBody[0], exitL, contL)
    elif s.elseBody.len > 0:
      ctx.indent.inc
      let elseTxt = ctx.printStmts(s.elseBody, exitL, contL)
      ctx.indent.dec
      result &= " else {\n" & elseTxt & '\n' & pad & '}'
  of sLoop:
    # loops print through printStmts, which derives the loop's own labels
    result = ""
  of sLab:
    # an exit label after a loop is the break target marker, dropped
    result = ""
  of sJmp:
    if s.jmpTarget == exitL and exitL.len > 0:
      result = pad & "break;"
    elif s.jmpTarget == contL and contL.len > 0:
      result = pad & "continue;"
    else:
      ctx.fail("unstructured jump to " & s.jmpTarget)
      result = ""
  of sScope:
    result = pad & "{\n"
    ctx.indent.inc
    result &= ctx.printStmts(s.scopeBody, exitL, contL)
    ctx.indent.dec
    result &= '\n' & pad & '}'
  of sAssume:
    # a loop-bound hint, HLSL has no assume construct
    result = ""
  of sRet:
    if s.retVal != nil:
      result = pad & "return " & ctx.printExpr(s.retVal) & ";"
    else:
      result = pad & "return;"
  of sDiscard:
    result = ""
  of sCall:
    var parts: seq[string]
    if s.callCallee.kind == eSym:
      parts.add s.callCallee.name
    else:
      parts.add ctx.printExpr(s.callCallee)
    for i, a in s.callArgs:
      if ctx.flavor == fvD3D12 and a.kind == eSym and
          ctx.bufferParams.contains(a.name):
        parts.add a.name
      else:
        parts.add ctx.spellCallArg(s.callCallee, i, a)
    if s.callCallee.kind == eSym and
        s.callCallee.synchroBuiltin == sbkThreadgroupBarrier:
      result = pad & "GroupMemoryBarrierWithGroupSync();"
    elif s.callCallee.kind == eSym and
        s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
      result = pad & "DeviceMemoryBarrierWithGroupSync();"
    else:
      result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"
  of sAsm:
    ctx.fail("inline assembly has no HLSL form")
    result = ""
  of sCase:
    ctx.fail("case statements are outside the CIRv2 closure, the frontend refuses them")
    result = ""

proc exprHasCall(e: CirExpr): bool =
  ## True when any subexpression of `e` is a call.
  ##
  ## A call may write memory. A conditional expression can evaluate
  ## both variants on some hardware, so only call-free variants may take
  ## the conditional form.
  if e == nil:
    return false
  if e.kind == eCall:
    return true
  for c in e.getExprChildren():
    if exprHasCall(c):
      return true

proc isSelectArm(s: CirStmt, name: string): bool =
  ## True when `s` assigns the local `name` exactly once from a plain expression,
  ## matching one variant of a hoisted if-expression.
  s.kind == sAsgn and s.lhs.kind == eSym and s.lhs.name == name and s.rhs != nil

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string =
  ## Prints one statement list. Plain statements take the caller's loop context,
  ## a `loop` derives its own labels: the exit label right after it and the continue label inside.
  var lines: seq[string]
  var i = 0
  while i < stmts.len:
    let s = stmts[i]
    var txt: string
    if s.kind == sLoop:
      let lExit = if i + 1 < stmts.len and stmts[i + 1].kind == sLab:
        stmts[i + 1].labName
      else:
        ""
      var lCont = ""
      for t in s.loopBody:
        if t.kind == sLab and t.labName.startsWith(ContinueLabelPrefix):
          lCont = t.labName
      ctx.indent.inc
      let bodyTxt = ctx.printStmts(s.loopBody, lExit, lCont)
      ctx.indent.dec
      txt = spaces(ctx.indent * 2) & "for (;;)\n" & spaces(ctx.indent * 2) &
        "{\n" & bodyTxt & "\n" & spaces(ctx.indent * 2) & '}'
      if lExit.len > 0:
        i.inc  # exit label, consumed
    elif s.kind == sVar and s.init == nil and
        s.declType.kind in {ctBool, ctInt, ctUint, ctFloat} and
        i + 1 < stmts.len and stmts[i + 1].kind == sIte and
        stmts[i + 1].thenBody.len == 1 and stmts[i + 1].elseBody.len == 1 and
        stmts[i + 1].thenBody[0].isSelectArm(s.name) and
        stmts[i + 1].elseBody[0].isSelectArm(s.name) and
        not exprHasCall(stmts[i + 1].cond) and
        not exprHasCall(stmts[i + 1].thenBody[0].rhs) and
        not exprHasCall(stmts[i + 1].elseBody[0].rhs):
      # a hoisted if-expression temp with one assignment per variant prints
      # as the conditional expression the source wrote
      ctx.indent.inc
      let condTxt = ctx.printExpr(stmts[i + 1].cond)
      let thenTxt = ctx.printExpr(stmts[i + 1].thenBody[0].rhs)
      let elseTxt = ctx.printExpr(stmts[i + 1].elseBody[0].rhs)
      ctx.indent.dec
      if ctx.failing():
        txt = ""
      else:
        txt = spaces(ctx.indent * 2) & ctx.spellHlslDecl(s.declType, s.name) &
          " = " & condTxt & " ? " & thenTxt & " : " & elseTxt & ";"
      i.inc  # consumes the ite with the temp it assigns
    else:
      txt = ctx.printStmt(s, exitL, contL)
    if txt.len > 0:
      lines.add txt
    i.inc
  return lines.join("\n")

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc spellResourceParam(ctx: var PrintCtx, p: CirParam, rw: bool): string =
  ## HLSL resource parameter of a buffer: StructuredBuffer / RWStructuredBuffer
  ## over the element type, `rw` picks the read-write form.
  let resTy = if rw: "RWStructuredBuffer" else: "StructuredBuffer"
  result = resTy & "<" & ctx.spellHlslScalar(p.typ.pElem) & "> " & p.name

proc classifyParams(ctx: var PrintCtx, fn: CirProc, graph: CallGraph, pushParams: var HashSet[string], buffers: var seq[CirParam]) =
  ## Splits one proc's parameters into buffer resources and values.
  ## - kernels: values join the Push block, buffers take register ordinals
  ## - device functions: values pass by value (a `var T` binds `inout`),
  ##   buffers are unbound resource params
  ## Rejects the parameter shapes the HLSL resource model cannot bind.
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      if p.typ.space != caDevice:
        ctx.fail("buffer parameter " & p.name & " in " & $p.typ.space &
          " space, buffers address device memory")
        return
      if ctx.flavor == fvVulkan:
        # element types are checked up front, the access emitter and the buffer-pointer typedef both read them
        discard ctx.bpElementStride(p.typ.pElem)
        if ctx.failing():
          return
      buffers.add p
    of ctPtr:
      if p.typ.space == caThread:
        if fn.isKernel:
          ctx.fail("by-reference parameter " & p.name & " in kernel `" &
            fn.name & "` has no binding, kernels take values and buffers")
          return
        # device function: the `inout` binding, reads and writes are direct
        discard
      else:
        ctx.fail("pointer parameter " & p.name & " in " & $p.typ.space &
          " space has no binding")
        return
    of ctArray:
      ctx.fail("array parameter " & p.name & " in `" & fn.name &
        "` has no binding, buffers pass as `ptr UncheckedArray[T]`")
      return
    of ctInt, ctUint:
      if p.typ.bits != 32:
        ctx.fail("64-bit integer parameter " & p.name & " in `" & fn.name &
          "` is rejected: int64_t is a parse error in the HLSL frontend and" &
          " uint64_t pulls the SPIR-V Int64 capability")
        return
      if fn.isKernel:
        pushParams.incl(p.name)
    of ctFloat:
      if p.typ.bits == 16:
        ctx.fail("float16 parameter " & p.name & " in `" & fn.name &
          "`, float16 types are outside the HLSL subset")
        return
      if fn.isKernel:
        pushParams.incl(p.name)
    of ctBool:
      if fn.isKernel:
        if ctx.flavor == fvVulkan:
          ctx.fail("bool parameter " & p.name & ", the fvVulkan push block has" &
            " no bool member")
          return
        pushParams.incl(p.name)
    of ctVoid, ctVector, ctStruct:
      if fn.isKernel:
        pushParams.incl(p.name)
    of ctProc:
      ctx.fail("procedure parameter " & p.name & " in `" & fn.name & "` has no binding")
      return

proc spellFnSignature(ctx: var PrintCtx, fn: CirProc, graph: CallGraph, pushParams: var HashSet[string], forwardDeclare: bool): string =
  ## `RetType name(params)` of one proc, buffers as resources, `var T` device parameters as `inout`.
  var buffers: seq[CirParam]
  ctx.classifyParams(fn, graph, pushParams, buffers)
  if ctx.failing():
    return ""
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      if ctx.flavor == fvVulkan:
        # address value only: reads and writes construct the buffer pointer per access
        params.add "uint64_t " & p.name
      else:
        let rw = graph.reached.contains(fn.name & "/" & p.name)
        params.add ctx.spellResourceParam(p, rw)
    of ctPtr:
      params.add "inout " & ctx.spellHlslDecl(p.typ.pElem, p.name)
    else:
      params.add ctx.spellHlslDecl(p.typ, p.name)
  if ctx.failing():
    return ""
  let ret = ctx.spellHlslScalar(fn.ret)
  if ctx.failing():
    return ""
  result = ret & ' ' & fn.name & '(' & params.join(", ") & ')'
  if forwardDeclare:
    result.add ';'

proc printFn(ctx: var PrintCtx, fn: CirProc, graph: CallGraph): string =
  ## One device function: signature, body, closing brace.
  ctx.fnName = fn.name
  ctx.indent = 0
  ctx.isKernel = false
  ctx.pushParams = initHashSet[string]()
  ctx.bufferParams = initHashSet[string]()
  ctx.rwBuffers = initHashSet[string]()
  ctx.addressParams = initTable[string, CirType]()
  for p in fn.params:
    if p.typ.kind == ctAptr:
      ctx.bufferParams.incl(p.name)
      ctx.addressParams[p.name] = p.typ
      if graph.reached.contains(fn.name & "/" & p.name):
        ctx.rwBuffers.incl(p.name)
  let bodyTxt = ctx.printStmts(fn.body, "", "")
  if ctx.failing():
    return ""
  var pushParams = initHashSet[string]()
  let sig = ctx.spellFnSignature(fn, graph, pushParams, forwardDeclare = false)
  if ctx.failing():
    return ""
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc registerStructRefs(ctx: var PrintCtx, t: CirType, refs: var HashSet[string]) =
  ## Records the struct type names reachable from `t`, through fields,
  ## array elements and vector elements.
  if t.isNil:
    return
  case t.kind
  of ctStruct:
    if t.name in refs:
      return
    refs.incl(t.name)
    for f in t.fields:
      ctx.registerStructRefs(f.typ, refs)
  of ctArray:
    ctx.registerStructRefs(t.aElem, refs)
  of ctVector:
    ctx.registerStructRefs(t.vElem, refs)
  else:
    discard

proc scanStmtStructRefs(ctx: var PrintCtx, ss: seq[CirStmt], refs: var HashSet[string]) =
  ## Records the struct types the statement tree references: declaration types,
  ## construction types and cast result types.
  for s in ss:
    case s.kind
    of sVar, sConst:
      ctx.registerStructRefs(s.declType, refs)
      if s.init != nil:
        if s.init.kind == eOconstr:
          ctx.registerStructRefs(s.init.ocType, refs)
        if s.init.kind == eCall:
          for a in s.init.args:
            if a.kind == eOconstr:
              ctx.registerStructRefs(a.ocType, refs)
    of sAsgn:
      if s.rhs != nil and s.rhs.kind == eOconstr:
        ctx.registerStructRefs(s.rhs.ocType, refs)
    of sIte:
      ctx.scanStmtStructRefs(s.thenBody, refs)
      ctx.scanStmtStructRefs(s.elseBody, refs)
    of sLoop:
      ctx.scanStmtStructRefs(s.loopBody, refs)
    of sScope:
      ctx.scanStmtStructRefs(s.scopeBody, refs)
    of sCall:
      for a in s.callArgs:
        if a.kind == eOconstr:
          ctx.registerStructRefs(a.ocType, refs)
    of sJmp, sLab, sAssume, sRet, sDiscard, sCase, sAsm:
      discard

proc spellStructDecls(ctx: var PrintCtx, refs: var HashSet[string],
    byName: Table[string, CirType], emitted: var HashSet[string],
    name: string, outp: var string) =
  ## Emits one struct definition, member structs first, deduplicated by name.
  ## HLSL resolves types at their declaration site, a referenced struct
  ## holding a pointer refuses, HLSL structs cannot take resources.
  if name in emitted or name notin refs:
    return
  let t = byName.getOrDefault(name, nil)
  if t == nil:
    ctx.fail("struct " & name & " is referenced but has no declaration")
    return
  if t.containsPtrDeep:
    ctx.fail("struct " & name & " holds a pointer, HLSL structs cannot hold" &
      " resources, the HLSL lowering pass dissolves pointer-holding locals")
    return
  emitted.incl name
  for f in t.fields:
    if f.typ.kind == ctStruct:
      ctx.spellStructDecls(refs, byName, emitted, f.typ.name, outp)
    elif f.typ.kind == ctArray:
      ctx.registerStructRefs(f.typ, refs)
      if f.typ.aElem.kind == ctStruct:
        ctx.spellStructDecls(refs, byName, emitted, f.typ.aElem.name, outp)
    elif f.typ.kind == ctVector:
      ctx.registerStructRefs(f.typ, refs)
  if ctx.failing():
    return
  outp.add "struct " & name & " {\n"
  if t.fields.len == 0:
    # HLSL requires at least one member in a struct
    outp.add "  uint _padding;\n"
  else:
    for f in t.fields:
      outp.add "  " & ctx.spellHlslDecl(f.typ, f.name) & ";\n"
  outp.add "};\n\n"

proc spellMainInputs(ctx: PrintCtx): string =
  ## Semantics for `main`, exactly the ones the kernel body reads.
  var inputs: seq[string]
  if cbkThreadPositionInGrid in ctx.coordReads:
    inputs.add "uint3 " & coordInputName(cbkThreadPositionInGrid) & " : SV_DispatchThreadID"
  if cbkThreadgroupPositionInGrid in ctx.coordReads:
    inputs.add "uint3 " & coordInputName(cbkThreadgroupPositionInGrid) & " : SV_GroupID"
  if cbkThreadPositionInThreadgroup in ctx.coordReads:
    inputs.add "uint3 " & coordInputName(cbkThreadPositionInThreadgroup) & " : SV_GroupThreadID"
  if cbkThreadIndexInThreadgroup in ctx.coordReads:
    inputs.add "uint " & coordInputName(cbkThreadIndexInThreadgroup) & " : SV_GroupIndex"
  result = inputs.join(", ")

proc printHlslModule*(m: CirModule, flavor: HlslFlavor = fvD3D12): tuple[src: string, err: string] =
  ## Emits the full HLSL translation unit: struct declarations, the binding section,
  ## the device function declarations and definitions, then `[numthreads]`,
  ## then the kernel as `main`.
  ##
  ## - fvD3D12: register resources and the `ConstantBuffer` push block
  ## - fvVulkan: the `[[vk::push_constant]]` struct with every member offset locked
  ##
  ## Returns "" as the error when clean, else the first refusal.
  when defined(cirv2TraceCir):
    writeFile("/tmp/cirv2dbg/hlsl_cir_dump.txt", renderModule(m))
  var ctx = PrintCtx(fnName: "<module>", flavor: flavor)
  var procsByName: Table[string, CirProc]
  for fn in m.procs:
    procsByName[fn.name] = fn
  ctx.procs = procsByName
  var kernel: CirProc = nil
  for fn in m.procs:
    if fn.isKernel:
      if kernel != nil:
        ctx.fail("exactly one kernel per module, found at least two")
        return ("", ctx.err)
      kernel = fn
  if kernel == nil:
    ctx.fail("exactly one kernel per module, found none")
    return ("", ctx.err)
  if kernel.ret.kind != ctVoid:
    ctx.fail("kernel `" & kernel.name & "` must return void, got " & $kernel.ret.kind)
    return ("", ctx.err)

  # call graph: read-write classification, then the recursion refusal
  var graph = CallGraph()
  for fn in m.procs:
    graph.procs[fn.name] = fn
  for fn in m.procs:
    graph.scanStoredStmts(fn, fn.body)
    graph.scanCallStmts(fn, fn.body)
  graph.classifyRwBuffers()
  ctx.checkDeviceFnRecursion(graph)
  if ctx.failing():
    return ("", ctx.err)

  # kernel resources: buffer parameters and Push members
  ctx.fnName = kernel.name
  var pushParams = initHashSet[string]()
  var buffers: seq[CirParam]
  ctx.classifyParams(kernel, graph, pushParams, buffers)
  if ctx.failing():
    return ("", ctx.err)

  # struct declarations for the referenced value structs, dependency order
  var refs = initHashSet[string]()
  for p in kernel.params:
    if p.name != "crucible_numworkgroups":
      ctx.registerStructRefs(p.typ, refs)
  for fn in m.procs:
    if fn.isKernel:
      continue
    ctx.registerStructRefs(fn.ret, refs)
    for p in fn.params:
      ctx.registerStructRefs(p.typ, refs)
    ctx.scanStmtStructRefs(fn.body, refs)
  ctx.scanStmtStructRefs(kernel.body, refs)
  # relocated const_mem declarations, struct-typed ones included
  ctx.scanStmtStructRefs(m.globals, refs)
  var byName: Table[string, CirType]
  for t in m.types:
    if t.kind == ctStruct:
      byName[t.name] = t
  var emitted = initHashSet[string]()
  var refList: seq[string]
  for name in refs:
    refList.add name
  for name in refList:
    ctx.spellStructDecls(refs, byName, emitted, name, result.src)
  if ctx.failing():
    return ("", ctx.err)

  # binding section: fvD3D12 register resources, fvVulkan push-constant struct with locked offsets
  if ctx.flavor == fvVulkan:
    # buffer-pointer aliases, one per element type in encounter order
    var bpEmitted: HashSet[string]
    for p in buffers:
      let nm = spellBpName(p.typ.pElem)
      if nm notin bpEmitted:
        bpEmitted.incl nm
        result.src &= "typedef vk::BufferPointer<" &
          ctx.spellHlslScalar(p.typ.pElem) & "> " & nm & ";\n"
    if bpEmitted.len > 0:
      result.src &= "\n"
    if kernel.params.len > 0:
      # one member per kernel parameter in host pack order, every member offset
      # locked with [[vk::offset]] to the launch-side packing, the grid member 16-byte aligned
      result.src &= "struct Push {\n"
      var off = 0
      for p in kernel.params:
        if p.typ.kind == ctAptr:
          off = roundUpMultiple(off, 8)
          result.src &= "  [[vk::offset(" & $off & ")]] uint64_t " & p.name & ";\n"
          off.inc 8
        elif p.name == "crucible_numworkgroups":
          # materialized dispatch grid size, an int3 the engine stages
          # on the 16-byte boundary the std430 vector rule gives it
          off = roundUpMultiple(off, 16)
          result.src &= "  [[vk::offset(" & $off & ")]] int3 " & p.name & ";\n"
          off.inc 12
        else:
          let (size, align) = ctx.pushMemberExtent(p.typ, p.name)
          if ctx.failing():
            return ("", ctx.err)
          off = roundUpMultiple(off, align)
          result.src &= "  [[vk::offset(" & $off & ")]] " &
            ctx.spellHlslDecl(p.typ, p.name) & ";\n"
          off.inc size
        if ctx.failing():
          return ("", ctx.err)
      result.src &= "};\n"
      result.src &= "[[vk::push_constant]] Push push;\n"
      result.src &= "\n"
  else:
    # one global register ordinal across the buffers, read-only `t`, read-write `u`, the Push block on `b`
    var ordinal = 0
    for p in buffers:
      let rw = graph.reached.contains(kernel.name & "/" & p.name)
      let reg = if rw: 'u' else: 't'
      result.src &= ctx.spellResourceParam(p, rw) & " : register(" & reg &
        $ordinal & ", space0);\n"
      ordinal.inc
    if pushParams.len > 0:
      result.src &= "struct Push {\n"
      for p in kernel.params:
        if p.name == "crucible_numworkgroups":
          # materialized dispatch grid size, an int3 the engine stages
          result.src &= "  int3 " & p.name & ";\n"
        elif p.name in pushParams:
          result.src &= "  " & ctx.spellHlslDecl(p.typ, p.name) & ";\n"
      result.src &= "};\n"
      result.src &= "ConstantBuffer<Push> push : register(b" & $ordinal &
        ", space0);\n"
      result.src &= "\n"
  if ctx.failing():
    return ("", ctx.err)

  # file-scope `static const` declarations relocated into `m.globals` by the HLSL lowering pass,
  # in encounter order: dxc folds them at compile time, they take no register
  # in the cbuffer layout
  var hoisted: seq[string]
  for s in m.globals:
    if s.kind == sVar and s.space == caConstant:
      if s.init == nil:
        ctx.fail("const_mem local " & s.name & " without an initializer")
      elif s.init.kind == eOconstr and s.declType.kind == ctStruct:
        hoisted.add "static const " & ctx.spellHlslDecl(s.declType, s.name) &
          " = " & ctx.printBraceList(s.declType, s.init) & ";"
      else:
        hoisted.add "static const " & ctx.spellHlslDecl(s.declType, s.name) &
          " = " & ctx.printExpr(s.init) & ";"
  for d in hoisted:
    result.src &= d & "\n"
  if hoisted.len > 0:
    result.src &= "\n"
  if ctx.failing():
    return ("", ctx.err)

  # forward declarations first, a call resolves to a definition emitted later through them
  var fns: seq[CirProc]
  for fn in m.procs:
    if not fn.isKernel:
      fns.add fn
  for fn in fns:
    ctx.fnName = fn.name
    var fnPush = initHashSet[string]()
    result.src &= ctx.spellFnSignature(fn, graph, fnPush, forwardDeclare = true) & '\n'
    if ctx.failing():
      return ("", ctx.err)
  if fns.len > 0:
    result.src &= "\n"
  for fn in fns:
    result.src &= ctx.printFn(fn, graph) & "\n\n"
  if ctx.failing():
    return ("", ctx.err)

  # body before signature: the printer must know which inputs the body
  # reads before it can write main's signature
  ctx.fnName = kernel.name
  ctx.indent = 0
  ctx.isKernel = true
  ctx.pushParams = pushParams
  ctx.bufferParams = initHashSet[string]()
  ctx.addressParams = initTable[string, CirType]()
  for p in buffers:
    ctx.bufferParams.incl(p.name)
    ctx.addressParams[p.name] = p.typ
  ctx.rwBuffers = initHashSet[string]()
  for p in buffers:
    if graph.reached.contains(kernel.name & "/" & p.name):
      ctx.rwBuffers.incl(p.name)
  ctx.workgroup = (if kernel.workgroup == [0, 0, 0]: [256, 1, 1]
                   else: kernel.workgroup)
  let bodyTxt = ctx.printStmts(kernel.body, "", "")
  if ctx.failing():
    return ("", ctx.err)
  result.src &= "[numthreads(" & $ctx.workgroup[0] & ", " & $ctx.workgroup[1] &
    ", " & $ctx.workgroup[2] & ")]\n"
  let inputs = ctx.spellMainInputs()
  result.src &= "void main(" & inputs & ")\n{\n" & bodyTxt & "\n}\n"
  return (result.src, "")
