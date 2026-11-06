# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, sets, strutils, tables]


import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import ../passes/pass_helpers
import ./print_common
import ./targets_registry
import ../../builtins/builtins_gpu_types
import ../../crucible_codegen
import ../../builtins/builtins_decl_helpers

# ############################################################
#
#            HLSL printer
#
# ############################################################
{.experimental: "codeReordering".}

type HlslFlavor* = enum
  ## Resource-binding flavor of the emitted HLSL.
  ## The flavor changes the resource-binding section only, the language body is identical.
  fvD3D12   ## D3D12-native StructuredBuffer / RWStructuredBuffer / ConstantBuffer
            ## on `register(tN/uN/bN, space0)`, consumable by DXC for DXIL and SPIR-V
  fvVulkan  ## zero-descriptor Vulkan flavor: every parameter binds into one `[[vk::push_constant]]` block,
            ## buffers as device addresses,
            ## accesses construct `vk::BufferPointer` from the address,
            ## dxc -spirv emits physical-storage-buffer words for the device-address binding


# ═══════════════════════════════════════════════════════════════════════
#  Per-proc emission state
# ═══════════════════════════════════════════════════════════════════════

type
  PrintCtx = object of WalkCtx
    ## Emission state of the HLSL walk: the inherited walk bookkeeping
    ## (proc name, refusal, indent, policy, literal and scalar token tables)
    ## plus the per-proc parameter classification below.
    isKernel: bool
    ## True while printing the entry kernel, coordinate builtins and push-constant reads bind the entry only.
    pushParams: HashSet[string]
    ## Push-constant parameters of the proc in flight, reads print through
    ## `push.<name>`, the printer refuses an assignment.
    bufferParams: HashSet[string]
    ## Buffer parameters of the proc in flight, the printer refuses a direct use (fvD3D12),
    ## the fvVulkan route reads the u64 device address.
    rwBuffers: HashSet[string]
    ## Buffer parameters classified read-write by the module write-scan,
    ## the printer refuses an assignment through a read-only one.
    flavor: HlslFlavor
    ## Resource-binding flavor in flight.
    addressParams: Table[string, CirType]
    ## Buffer parameters of the proc in flight by name to their `aptr` type:
    ## the fvVulkan access emitter reads the element type per access.
    coordReads: HashSet[CirCoordBuiltinKind]
    ## Canonical coordinate builtin names that the kernel body reads.
    ## The entry head declares their inputs in `CirCoordBuiltinKind` declaration order.
    procs: Table[string, CirProc]
    ## Module procs by name, by-reference call arguments read callee
    ## parameter types here.
    graph: CallGraph
    ## Stored call-graph record (`CirModule.callGraph`): the read-write classification of every buffer parameter.
    kernelPush: HashSet[string]
    ## Push-constant parameter names of the module's kernel, from `classifyParams`.
    kernelBuffers: seq[CirParam]
    ## Buffer parameters of the module's kernel, from the same classification.

const printerTarget = "hlsl"

const hlslVocab = BuiltinVocab(
  shuffleText: makeTexts([
    (cbkSimdShuffle, "WaveReadLaneAt($1, $2)")]),
  stmtText: makeTexts([
    (cbkThreadgroupBarrier, "GroupMemoryBarrierWithGroupSync()"),
    (cbkThreadgroupBarrierDevice, "DeviceMemoryBarrierWithGroupSync()")]))

const hlslEmit = TargetEmit(
  vocab: hlslVocab,
  functionalCtor: false, ctorOpen: "{ ", ctorClose: " }")


# ═══════════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════════

# Shared-walk mixin targets, declared so the overlay procs are module-scope
# symbols when the walk's generics resolve (mixin ignores codeReordering).
proc emitTypeName(ctx: var PrintCtx, t: CirType): string
proc coordRef(ctx: var PrintCtx, e: CirExpr): string
proc coordDot(ctx: var PrintCtx, e: CirExpr): string
proc printConv(ctx: var PrintCtx, e: CirExpr): string
proc printSizeof(ctx: var PrintCtx, e: CirExpr): string
proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string
proc getSymbolText(ctx: var PrintCtx, e: CirExpr): string
proc convBuiltinText(ctx: var PrintCtx, call: CirExpr, args: seq[string]): string
proc printOconstr(ctx: var PrintCtx, e: CirExpr): string
proc printAconstr(ctx: var PrintCtx, e: CirExpr): string
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string
proc getInitText(ctx: var PrintCtx, s: CirStmt): string
proc printCallArgTexts(ctx: var PrintCtx, callee: CirExpr, args: seq[CirExpr]): seq[string]
proc printAsgn(ctx: var PrintCtx, s: CirStmt): string
proc prepProc(ctx: var PrintCtx, fn: CirProc)
proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string
proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string
proc printParamBindings(ctx: var PrintCtx, kernels: seq[CirProc]): string
proc printPrelude(ctx: var PrintCtx, m: CirModule): string

proc refuseHlslScalarName(ctx: var PrintCtx, t: CirType): void =
  ## Refusal side of the type-name lookup, the width exists in Nim, HLSL names only the subset.
  case t.kind
  of ctInt:
    ctx.fail("int" & $t.bits & " has no HLSL type name")
  of ctUint:
    ctx.fail("uint" & $t.bits & " has no HLSL type name")
  of ctFloat:
    ctx.fail("float16 has no HLSL type name, float16 types are outside the HLSL subset")
  of ctBf16:
    ctx.fail("bfloat16 has no HLSL type name, bfloat16 types are outside the HLSL subset")
  else:
    discard

proc emitHlslScalar(ctx: var PrintCtx, t: CirType): string =
  ## HLSL name of a non-array type, "" when the caller builds the type name.
  case t.kind
  of ctVoid, ctBool:
    result = getScalarName(TargetVocabs[cgHlsl].scalarNames, t)
  of ctInt, ctUint, ctFloat:
    result = getScalarName(TargetVocabs[cgHlsl].scalarNames, t)
    if result.len == 0:
      refuseHlslScalarName(ctx, t)
  of ctBf16:
    refuseHlslScalarName(ctx, t)
  of ctVector:
    let elem = emitHlslScalar(ctx, t.vElem)
    result = getVectorName(hlslVectorRule, TargetVocabs[cgHlsl].scalarNames, t)
    if result.len == 0:
      ctx.fail("vector of " & (if elem.len > 0: elem else: $t.vElem.kind) & "x" & $t.vN & " has no HLSL type name, only 2..4-wide float/int/uint/bool vectors have a name")
  of ctStruct:
    result = t.name
  of ctString:
    ctx.fail("string has no HLSL type name")
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc emitHlslTypeBare(ctx: var PrintCtx, t: CirType): string =
  case t.kind
  of ctArray:
    ctx.fail("array type has no bare HLSL name, arrays compose at their declaration")
  else:
    result = emitHlslScalar(ctx, t)

proc emitHlslDecl(ctx: var PrintCtx, t: CirType, ident: string): string =
  if t.kind == ctArray:
    result = emitHlslScalar(ctx, t.aElem) & ' ' & ident & "[" & $t.aN & "]"
  else:
    result = emitHlslScalar(ctx, t) & ' ' & ident

proc emitBpName(t: CirType): string =
  case t.kind
  of ctInt: result = "BpI32"
  of ctUint: result = "BpU32"
  of ctFloat: result = (if t.bits == 64: "BpF64" else: "BpF32")
  else: result = ""

proc bpElementStrideCheck*(elem: CirType): tuple[stride: int, err: string] =
  ## Returns the byte stride of one buffer element on the fvVulkan route and the refusal when the element has no binding.
  ## `err` is "" when the element binds, else the refusal detail. Readers: the fvVulkan access emitter, the parameter-classification pass.
  case elem.kind
  of ctInt, ctUint:
    if elem.bits != 32:
      return (0, "buffer element of " & $elem.bits & "-bit integer has no fvVulkan" &
        " binding, the element envelope is 32-bit integers, float and double")
    result = (4, "")
  of ctFloat:
    if elem.bits != 32 and elem.bits != 64:
      return (0, "buffer element of float" & $elem.bits & " has no fvVulkan" &
        " binding, the element envelope is 32-bit integers, float and double")
    result = (elem.bits div 8, "")
  else:
    result = (0, "buffer element of " & $elem.kind & " type has no fvVulkan binding," &
      " the element envelope is 32-bit integers, float and double")

proc pushMemberExtent(ctx: var PrintCtx, t: CirType, label: string): tuple[size, align: int] =
  ## Packed size and alignment of one fvVulkan push-block value member.
  ## Member offsets match the launch-side packing: scalars and vectors are
  ## width-aligned, struct members pack flat over scalars, the alignment is
  ## the max field width, the size rounds up to it.
  case t.kind
  of ctInt, ctUint, ctFloat:
    let w = t.bits div 8
    result = (w, w)
  of ctBool:
    ctx.fail("bool push member " & label & ", the fvVulkan push block has no bool member")
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
        ctx.fail("field " & f.name & " of push member " & label & " is a " & $f.typ.kind & ", the fvVulkan push block packs 32/64-bit scalars flat")
    size = roundUpMultiple(size, result.align)
    result.size = size
  else:
    ctx.fail("push member " & label & " of " & $t.kind & " type has no fvVulkan binding")

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  ctx.emitHlslTypeBare(t)

proc coordRef(ctx: var PrintCtx, e: CirExpr): string =
  let rej = getCoordReadRejection(ctx, e.coordBuiltin)
  if rej.len > 0:
    ctx.fail(rej)
    return ""
  if e.coordBuiltin == cbkThreadIndexInThreadgroup:
    ctx.coordReads.incl(e.coordBuiltin)
    return "int(" & getCoordInputName(e.coordBuiltin) & ")"
  ctx.fail("whole-vector read of a coordinate builtin is outside the HLSL subset")
  ""

proc coordDot(ctx: var PrintCtx, e: CirExpr): string =
  let rej = getCoordReadRejection(ctx, e.dotBase.coordBuiltin)
  if rej.len > 0:
    ctx.fail(rej)
    return ""
  ctx.emitCoordComponent(e.dotBase.coordBuiltin, e.fieldName)

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  return emitHlslTypeBare(ctx, e.castType) & "(" & ctx.printExpr(e.castOperand) & ")"

proc printSizeof(ctx: var PrintCtx, e: CirExpr): string =
  ctx.fail("sizeof has no HLSL emission, HLSL struct layout differs from the buffer layout")
  ""

proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## `eAt`/`ePat` text by base:
  ## - address parameters index by the element stride
  ## - buffer parameters index by subscript, arrays through the shared composer
  ## - any other base errors out
  let base = if e.kind == eAt: e.atBase else: e.patBase
  let idx = if e.kind == eAt: e.atIndex else: e.patIndex
  let idxTxt = ctx.printExpr(idx)
  if ctx.flavor == fvVulkan and base.kind == eSym and
      ctx.addressParams.hasKey(base.name):
    let elem = ctx.addressParams[base.name].pElem
    # address arithmetic indexes buffers by this stride
    let (stride, strideErr) = bpElementStrideCheck(elem)
    if strideErr.len > 0:
      ctx.fail(strideErr)
      return ""
    if ctx.failing():
      return ""
    return emitBpName(elem) & "(" & ctx.printExpr(base) &
      " + (uint64_t)(" & idxTxt & ") * " & $stride & ").Get()"
  if base.kind == eSym and ctx.bufferParams.contains(base.name):
    return base.name & "[" & idxTxt & "]"
  let bt = getExprType(base)
  if bt != nil and bt.kind == ctArray:
    let (baseTxt, _) = ctx.printSubscriptParts(e)
    return baseTxt & "[" & idxTxt & "]"
  ctx.fail("indexing an " & (if bt == nil: "untyped" else: $bt.kind) & " base is outside the HLSL subset")
  ""

proc getSymbolText(ctx: var PrintCtx, e: CirExpr): string =
  ## Symbol read text. Push-constant parameters read through `push.<name>`.
  ## A buffer parameter is its u64 device address on fvVulkan, a direct use errors out on fvD3D12.
  if ctx.pushParams.contains(e.name):
    return "push." & e.name
  if ctx.bufferParams.contains(e.name):
    if ctx.flavor == fvVulkan:
            return (if ctx.isKernel: "push." & e.name else: e.name)
    ctx.fail("bare pointer value use of buffer parameter " & e.name)
    return ""
  e.name

proc convBuiltinText(ctx: var PrintCtx, call: CirExpr, args: seq[string]): string =
  ## dxc rejects what it cannot parse.
  ""

proc printCallArg(ctx: var PrintCtx, callee: CirExpr, i: int, a: CirExpr): string =
  ## One call argument: a buffer resource passes by name, each call site supplies
  ## the resource bound to its own register. An address-of over a variable argument
  ## to a by-reference parameter emits the variable itself, the `inout` binding
  ## accepts lvalues directly.
  if ctx.flavor == fvD3D12 and a.kind == eSym and
      ctx.bufferParams.contains(a.name):
    return a.name
  if a.kind == eAddr and a.addrOperand.kind == eSym and
      callee.kind == eSym and ctx.procs.hasKey(callee.name):
    let fn = ctx.procs[callee.name]
    if i < fn.params.len and fn.params[i].typ.kind == ctPtr and
        fn.params[i].typ.addressSpace == caThread:
      return a.addrOperand.name
  return ctx.printExpr(a)


proc printCallArgTexts(ctx: var PrintCtx, callee: CirExpr, args: seq[CirExpr]): seq[string] =
  for i, a in args:
    result.add ctx.printCallArg(callee, i, a)

proc printOconstr(ctx: var PrintCtx, e: CirExpr): string =
  ## DXC accepts a struct brace list only in declaration initializers.
  if e.ocType.kind == ctVector:
    var parts: seq[string]
    for kvv in e.kvFields:
      parts.add ctx.printExpr(kvv.value)
    return ctx.emitHlslScalar(e.ocType) & "(" & parts.join(", ") & ")"
  ctx.printBraceList(e.ocType, e)

proc printAconstr(ctx: var PrintCtx, e: CirExpr): string =
  ## A braced list is the array-initializer form DXC accepts.
  var parts: seq[string]
  for el in e.elems:
    parts.add ctx.printExpr(el)
  "{ " & parts.join(", ") & " }"

proc getCoordReadRejection(ctx: PrintCtx, k: CirCoordBuiltinKind): string =
  if not ctx.isKernel:
    return "coordinate semantics bind the kernel entry point only"
  if k == cbkThreadgroupsPerGrid:
    return "threadgroups_per_grid has no HLSL semantic"
  return ""

# ═══════════════════════════════════════════════════════════════════════
#  Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc getCoordInputName(k: CirCoordBuiltinKind): string =
  case k
  of cbkThreadPositionInGrid: "gid"
  of cbkThreadgroupPositionInGrid: "groupId"
  of cbkThreadPositionInThreadgroup: "groupThreadId"
  of cbkThreadIndexInThreadgroup: "gidx"
  else: ""

proc getCoordEntryInput(k: CirCoordBuiltinKind): string =
  case k
  of cbkThreadPositionInGrid: "uint3 gid : SV_DispatchThreadID"
  of cbkThreadgroupPositionInGrid: "uint3 groupId : SV_GroupID"
  of cbkThreadPositionInThreadgroup: "uint3 groupThreadId : SV_GroupThreadID"
  of cbkThreadIndexInThreadgroup: "uint gidx : SV_GroupIndex"
  else: ""

proc orderedCoordReads(reads: HashSet[CirCoordBuiltinKind]): seq[CirCoordBuiltinKind] =
  for k in cbkThreadPositionInGrid..cbkThreadIndexInThreadgroup:
    if k in reads:
      result.add k

const hlslEntryInputs*: EntryInputs = block:
  var inputs: EntryInputs
  for k in cbkThreadPositionInGrid..cbkThreadIndexInThreadgroup:
    let text = getCoordEntryInput(k)
    if text.len > 0:
      inputs[k] = text
  inputs

proc emitCoordComponent(ctx: var PrintCtx, k: CirCoordBuiltinKind, field: string): string =
  ## One coordinate-builtin component, cast to the canonical `int` type.
  ctx.coordReads.incl(k)
  case k
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
    result = "int(" & getCoordInputName(k) & ")"
  else:
    result = "int(" & getCoordInputName(k) & "." & field & ")"

# ═══════════════════════════════════════════════════════════════════════
#  Expressions
# ═══════════════════════════════════════════════════════════════════════

proc constructionLeaves(ctx: var PrintCtx, outp: var seq[tuple[p, txt: string]], t: CirType, v: CirExpr, path: string): bool =
  ## Appends one (dotted path, leaf text) entry per bound leaf field.
  ##
  ## - `path` prefixes each leaf path, "" at the top
  ## - a zero-field struct contributes no entry, its padding member
  ##   zero-initializes
  ## - returns false once the walk recorded a refusal
  for f in t.fields:
    var val: CirExpr = nil
    for kvv in v.kvFields:
      if kvv.name == f.name:
        val = kvv.value
    if val == nil:
      ctx.fail("field " & f.name & " of " & t.name & " is not bound by the construction")
      return false
    let leafPath = if path.len == 0: f.name else: path & "." & f.name
    if val.kind == eOconstr and f.typ.kind == ctStruct:
      if not ctx.constructionLeaves(outp, f.typ, val, leafPath):
        return false
    else:
      outp.add (leafPath, ctx.printExpr(val))
  return true

proc emitHlslInit(ctx: var PrintCtx, t: CirType, v: CirExpr): string =
  if v != nil and v.kind == eOconstr and t.kind == ctStruct:
    return ctx.printBraceList(t, v)
  return ctx.printExpr(v)

proc printBraceList(ctx: var PrintCtx, t: CirType, v: CirExpr): string =
  ## Flat brace list of one object construction: `{ 0, 1, 2 }` at any depth.
  ## A zero-field struct contributes one `0` slot for its padding member.
  var leaves: seq[tuple[p, txt: string]]
  if not ctx.constructionLeaves(leaves, t, v, "") or ctx.failing():
    return ""
  if t.fields.len == 0:
    return "{ 0 }"
  result = "{ "
  var texts: seq[string]
  for l in leaves:
    texts.add l.txt
  result.add texts.join(", ")
  result.add " }"

# ═══════════════════════════════════════════════════════════════════════
#  Statements
# ═══════════════════════════════════════════════════════════════════════
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## Local-space policy: buffers are kernel parameters, shared and device spaces never reach the printer.
  if s.declType.kind == ctAptr:
    ctx.fail("buffer local " & s.name & " has no HLSL binding, buffers are kernel parameters")
    return false
  if s.addressSpace == caThreadgroup:
    ctx.fail("local " & s.name & " in threadgroup addressSpace, HLSL has no program-scope shared emission")
    return false
  if s.addressSpace == caConstant:
    ctx.fail("local " & s.name & " in constant addressSpace, the HLSL lowering pass moves const_mem locals to program scope")
    return false
  if s.addressSpace == caDevice:
    ctx.fail("local " & s.name & " in device addressSpace, HLSL locals are per-thread registers")
    return false
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  ctx.emitHlslDecl(s.declType, s.name)

proc getInitText(ctx: var PrintCtx, s: CirStmt): string =
  ctx.emitHlslInit(s.declType, s.init)

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment.
  let pad = spaces(ctx.indent * 2)
  let root = exprRootSym(s.lhs)
  if root != nil and ctx.bufferParams.contains(root.name):
    if ctx.flavor == fvD3D12 and root.name notin ctx.rwBuffers:
      ctx.fail("assignment through read-only buffer parameter " & root.name)
      return ""
    result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"
    return
  if s.lhs.kind == eSym and ctx.pushParams.contains(s.lhs.name):
    ctx.fail("assignment to push-constant parameter " & s.lhs.name & ", push constants are read-only inputs")
    return ""
  if s.rhs != nil and s.rhs.kind == eOconstr and s.rhs.ocType.kind == ctStruct:
    # DXC rejects compound literals on assignment, member-wise leaves are the form it accepts
    var leaves: seq[tuple[p, txt: string]]
    if not ctx.constructionLeaves(leaves, s.rhs.ocType, s.rhs, "") or ctx.failing():
      return ""
    var lines: seq[string]
    let target = ctx.printExpr(s.lhs)
    for l in leaves:
      lines.add target & "." & l.p & " = " & l.txt & ";"
    result = lines.join("\n")
    return
  if s.rhs != nil and s.rhs.kind == eAconstr:
    ctx.fail("array constructor assignment has no HLSL form")
    return ""
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

# ═══════════════════════════════════════════════════════════════════════
#  Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitResourceParam(ctx: var PrintCtx, p: CirParam, rw: bool): string =
  let resTy = if rw: "RWStructuredBuffer" else: "StructuredBuffer"
  result = resTy & "<" & ctx.emitHlslScalar(p.typ.pElem) & "> " & p.name

proc printProcHead(ctx: var PrintCtx, fn: CirProc): string =
  ## Head of one proc, signature and open brace:
  ##
  ## - a device function: the direct `RetType name(params)` signature, buffers as resources, `var T` device parameters as `inout`
  ##   an `{.inline.}` proc's signature opens with the `inline ` qualifier
  ## - the classification pass refuses illegal parameters, and the printer reads the stored records
  ## - the entry kernel: the `[numthreads(...)]` attribute line, then `void <entry>(<entry inputs>)` over the body-walk semantics
  ##   exportc renames the entry for the host runtime lookup, Vulkan ingest reads the entry name back from the `[numthreads]` line
  ## - the kernel's open brace prints on its own line (HLSL's brace style) and the body and the closing brace are the caller's
  if fn.isKernel:
    let wg = getWorkgroup(fn)
    result = "[numthreads(" & $wg[0] & ", " & $wg[1] & ", " & $wg[2] & ")]\n"
    # a kernel with no explicit name prints as `main`, device functions print under the identifier that their call sites reference
    let entryName = if fn.exportName.len > 0: fn.exportName else: "main"
    result &= "void " & entryName & "(" &
      emitEntryInputs(orderedCoordReads(ctx.coordReads), hlslEntryInputs) & ")\n{\n"
    return
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      if ctx.flavor == fvVulkan:
        # address value only: reads and writes construct the buffer pointer per access
        params.add "uint64_t " & p.name
      else:
        let rw = ctx.graph.reached.contains(fn.name & "/" & p.name)
        params.add ctx.emitResourceParam(p, rw)
    of ctPtr:
      params.add "inout " & ctx.emitHlslDecl(p.typ.pElem, p.name)
    else:
      params.add ctx.emitHlslDecl(p.typ, p.name)
  let ret = ctx.emitHlslScalar(fn.ret)
  if fn.inline:
    result = "inline "
  result &= ret & ' ' & fn.name & '(' & params.join(", ") & ')'

proc classifyParams(ctx: var PrintCtx, pushParams: var HashSet[string], buffers: var seq[CirParam], fn: CirProc) =
  ## Splits one proc's parameters into buffer resources and values:
  ##
  ## - kernels: values pack into the Push block, buffer parameters map to register ordinals
  ## - device functions: values pass by value with a `var T` parameter as `inout`, buffers are unbound resource params
  ##
  ## Classification runs at the signature-emission sites, so a module printed
  ## without the pipeline classifies as one printed through it.
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      if p.typ.addressSpace != caDevice:
        ctx.fail("buffer parameter " & p.name & " in " & $p.typ.addressSpace & " addressSpace, buffers address device memory")
        return
      if ctx.flavor == fvVulkan:
        # element types are checked up front, the access emitter and the buffer-pointer typedef both read them
        let (_, err) = bpElementStrideCheck(p.typ.pElem)
        if err.len > 0:
          ctx.fail(err)
          return
      buffers.add p
    of ctString:
      # HLSL has no string type name, the scalar-name lookup refuses it
      discard
    of ctPtr:
      if p.typ.addressSpace == caThread:
        if fn.isKernel:
          ctx.fail("by-reference parameter " & p.name & " in kernel `" & fn.name & "` has no binding")
          return
        # device function: the `inout` binding, reads and writes are direct
        discard
      else:
        ctx.fail("pointer parameter " & p.name & " in " & $p.typ.addressSpace & " addressSpace has no binding")
        return
    of ctArray:
      ctx.fail("array parameter " & p.name & " in `" & fn.name & "` has no binding, buffers pass as `ptr UncheckedArray[T]`")
      return
    of ctInt, ctUint, ctFloat, ctBool, ctVoid, ctVector, ctStruct, ctSpan, ctBf16:
      # spans split to pointer + length in the shared lowering, none reaches the printer
      if fn.isKernel:
        pushParams.incl(p.name)
    of ctProc:
      ctx.fail("procedure parameter " & p.name & " in `" & fn.name & "` has no binding")
      return

proc prepProc(ctx: var PrintCtx, fn: CirProc) =
  ## Per-proc state of the HLSL walk:
  ##
  ## - the entry kernel: push-constant parameters from the stored classification record, buffer parameters to their resource or address route
  ## - a device function: its own buffer parameters, read-write classification from the stored call-graph record
  ## - the kernel's workgroup sizes, the source of the `threads_per_threadgroup` component tokens
  ctx.isKernel = fn.isKernel
  ctx.pushParams = initHashSet[string]()
  ctx.bufferParams = initHashSet[string]()
  ctx.rwBuffers = initHashSet[string]()
  ctx.addressParams = initTable[string, CirType]()
  var buffers: seq[CirParam]
  if fn.isKernel:
    ctx.pushParams = ctx.kernelPush
    ctx.workgroup = getWorkgroup(fn)
    buffers = ctx.kernelBuffers
  else:
    # a device function's values pass by value, the push half records nothing
    var devicePush: HashSet[string]
    ctx.classifyParams(devicePush, buffers, fn)
    if ctx.failing():
      return
  for p in buffers:
    ctx.bufferParams.incl(p.name)
    ctx.addressParams[p.name] = p.typ
    if ctx.graph.reached.contains(fn.name & "/" & p.name):
      ctx.rwBuffers.incl(p.name)

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## dxc folds static const storage at compile time, no cbuffer register is reserved.
  if s.init == nil:
    ctx.refuseUninitializedConstMem(s.name)
    return ""
  return printDeclStmt("", "static const " & ctx.emitHlslDecl(s.declType, s.name), ctx.emitHlslInit(s.declType, s.init))

const hlslModuleShape = ModuleShape(
  spaces: {caConstant},
  structTypedef: false,
  structPadding: "uint _padding",
  structFieldTerm: ';',
  structClose: "};",
  fwdDecls: true,
  moduleOrderProcs: false,
  kernelSep: "\n",
  customProcHead: true,
  printsRefBlocks: false,
  printsParamBindings: true,
  printsReduceTemplates: false)

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  if f.typ.containsPtrDeep:
    ctx.fail("struct " & t.name & " holds a pointer, HLSL structs cannot hold resources")
    return ""
  ctx.emitHlslDecl(f.typ, f.name)

proc printParamBindings(ctx: var PrintCtx, kernels: seq[CirProc]): string =
  ## Resource-binding section of the module's one kernel, per flavor:
  ##
  ## - fvD3D12: register resources on one global ordinal, read-only `t`, read-write `u`, then the Push block on `b`
  ## - fvVulkan: buffer-pointer aliases, then the `[[vk::push_constant]]` struct with every member offset locked to the launch-side packing
  if ctx.flavor == fvVulkan:
    var bpEmitted: HashSet[string]
    for p in ctx.kernelBuffers:
      let nm = emitBpName(p.typ.pElem)
      if nm notin bpEmitted:
        bpEmitted.incl nm
        result &= "typedef vk::BufferPointer<" &
          ctx.emitHlslScalar(p.typ.pElem) & "> " & nm & ";\n"
    if bpEmitted.len > 0:
      result &= "\n"
    if kernels[0].params.len > 0:
      result &= "struct Push {\n"
      var off = 0
      for p in kernels[0].params:
        if p.typ.kind == ctAptr:
          off = roundUpMultiple(off, 8)
          result &= "  [[vk::offset(" & $off & ")]] uint64_t " & p.name & ";\n"
          off.inc 8
        elif p.name == "crucible_numworkgroups":
          # materialized dispatch grid size, an int3 the engine stages
          # on the 16-byte boundary the std430 vector rule gives it
          off = roundUpMultiple(off, 16)
          result &= "  [[vk::offset(" & $off & ")]] int3 " & p.name & ";\n"
          off.inc 12
        else:
          let (size, align) = ctx.pushMemberExtent(p.typ, p.name)
          if ctx.failing():
            return ""
          off = roundUpMultiple(off, align)
          result &= "  [[vk::offset(" & $off & ")]] " &
            ctx.emitHlslDecl(p.typ, p.name) & ";\n"
          off.inc size
        if ctx.failing():
          return ""
      result &= "};\n"
      result &= "[[vk::push_constant]] Push push;\n"
      result &= "\n"
  else:
    var ordinal = 0
    for p in ctx.kernelBuffers:
      let rw = ctx.graph.reached.contains(kernels[0].name & "/" & p.name)
      let reg = if rw: 'u' else: 't'
      result &= ctx.emitResourceParam(p, rw) & " : register(" & reg &
        $ordinal & ", space0);\n"
      ordinal.inc
    if ctx.kernelPush.len > 0:
      result &= "struct Push {\n"
      for p in kernels[0].params:
        if p.name == "crucible_numworkgroups":
          # materialized dispatch grid size, an int3 the engine stages
          result &= "  int3 " & p.name & ";\n"
        elif p.name in ctx.kernelPush:
          result &= "  " & ctx.emitHlslDecl(p.typ, p.name) & ";\n"
      result &= "};\n"
      result &= "ConstantBuffer<Push> push : register(b" & $ordinal &
        ", space0);\n"
      result &= "\n"

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  ""

proc printHlslModule*(m: CirModule, flavor: HlslFlavor = fvD3D12): tuple[src: string, err: string] =
  ## Emits the full HLSL translation unit through the shared module skeleton, returning "" as the error when clean, else the first refusal.
  var ctx = initPrintCtx(hlslWalkPolicy, hlslVectorRule, hlslEmit, cgHlsl)
  ctx.flavor = flavor
  ctx.procs = initTable[string, CirProc]()
  for fn in m.procs:
    ctx.procs[fn.name] = fn
  let kernel = getEntryKernel(ctx, m)
  if ctx.failing():
    return ("", ctx.err)
  if kernel.ret.kind != ctVoid:
    ctx.fail("kernel `" & kernel.name & "` must return void, got " & $kernel.ret.kind)
    return ("", ctx.err)
  ctx.graph = m.callGraph
  ctx.fnName = kernel.name
  ctx.classifyParams(ctx.kernelPush, ctx.kernelBuffers, kernel)
  if ctx.failing():
    return ("", ctx.err)
  printModule(ctx, m, @[kernel], hlslModuleShape)

proc codegen*(m: CirModule, flavor: HlslFlavor = fvD3D12): string =
  codegenModule(printHlslModule(m, flavor))
