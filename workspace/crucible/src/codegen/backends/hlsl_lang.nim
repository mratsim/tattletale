# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## HLSL printer: a structured CIRv2 module rendered as HLSL compute source for DXC,
## table-driven over the closed set of type kinds.
## Two resource-binding flavors share one language body, the binding section differs.
##
## Emission contract:
## - one `{.global.}` kernel per module, emitted as `main` unless
##   `{.exportc: "name".}` renames the entry,
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
## - the parameter-classification analysis pass stores every proc's parameter
##   classification in `CirModule.paramClasses`, the printer reads the records
## - a whole-vector read of a 3-wide coordinate builtin refuses, components read
##   one by one, the scalar `thread_index_in_threadgroup` reads whole as `int(gidx)`
## - the dispatch grid size has no HLSL semantic: a `threadgroups_per_grid` read refuses,
##   the HLSL lowering pass materializes the read into the trailing `crucible_numworkgroups` parameter,
##   staged at launch as a descriptor-route uniform on fvD3D12, as a push block member on fvVulkan
##
## Addressing, memory classes, jumps:
## - HLSL has no pointer values: pointer locals, pointer-holding struct locals,
##   `var` parameter derefs all unwrap in the HLSL lowering pass,
##   anything pointer shaped reaching the printer refuses
## - `{.const_mem.}` locals relocate into `CirModule.globals` through the HLSL lowering pass,
##   file-scope `static const` output, a smem, device-space or surviving
##   constant-space local refuses
## - the fp16/bf16/fp8 conversion builtins refuse, half-width types sit outside the HLSL subset
##
## Statements and constructions:
## - a loop prints `while (cond)`, a conditionless loop prints the braceless
##   `for (;;)` head with the brace on its own line, `{.unroll.}` adds `[unroll]`
## - object constructions emit flat brace lists at any depth, `{ 0, 1, 2 }`,
##   the form DXC accepts over uniformly scalar members, a constructor assignment
##   lowers to member-wise leaves, DXC rejects compound literals on assignment.

import std/[macros, sets, strutils, tables]

import ../ir/ast_datatypes

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import ../passes/pass_helpers
import ./print_common
import ./targets_rows

type HlslFlavor* = enum
  ## Resource-binding flavor of the emitted HLSL.
  ## The flavor changes the resource-binding section only, the language body is identical.
  fvD3D12   ## D3D12-native StructuredBuffer / RWStructuredBuffer / ConstantBuffer
            ## on `register(tN/uN/bN, space0)`, consumable by DXC for DXIL and SPIR-V
  fvVulkan  ## zero-descriptor Vulkan flavor: every parameter binds into one `[[vk::push_constant]]` block,
            ## buffers as device addresses,
            ## accesses construct `vk::BufferPointer` from the address,
            ## dxc -spirv emits physical-storage-buffer words for the address route


# ═══════════════════════════════════════════════════════════════════════
# Per-proc emission state
# ═══════════════════════════════════════════════════════════════════════

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    policy: KindRow
    ## Kind row of the target. The shared walk reads its policy fields.
    lit: LitTokens
    ## Literal token table of the target, the shared literal head reads it.
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

const printerTarget = "hlsl"

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitHlslScalar(ctx: var PrintCtx, t: CirType): string =
  ## HLSL name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid, ctBool:
    result = scalarName(hlslScalarRow(), t)
  of ctInt, ctUint, ctFloat:
    result = scalarName(hlslScalarRow(), t)
    if result.len == 0:
      case t.kind
      of ctInt:
        ctx.fail("int" & $t.bits & " has no HLSL type name: the 8/16-bit forms" &
          " are parse errors and int64_t is a parse error in the HLSL frontend," &
          " 64-bit integers are rejected")
      of ctUint:
        ctx.fail("uint" & $t.bits & " has no HLSL type name: the 8/16-bit forms" &
          " are parse errors and uint64_t pulls the SPIR-V Int64 capability," &
          " 64-bit integers are rejected")
      else:
        ctx.fail("float16 has no HLSL type name, float16 types are outside the HLSL subset")
  of ctVector:
    let elem = emitHlslScalar(ctx, t.vElem)
    result = vectorName(hlslVectorRule(), hlslScalarRow(), t)
    if result.len == 0:
      ctx.fail("vector of " & (if elem.len > 0: elem else: $t.vElem.kind) &
        "x" & $t.vN & " has no HLSL type name, only 2..4-wide float/int/uint/bool vectors have a name")
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc emitHlslTypeBare(ctx: var PrintCtx, t: CirType): string =
  ## HLSL name of `t` with no identifier, the form of casts and constructors.
  ## Arrays have no bare form, `T name[N]` composes at the declaration.
  case t.kind
  of ctArray:
    ctx.fail("array type has no bare HLSL name, arrays compose at their declaration")
  else:
    result = emitHlslScalar(ctx, t)

proc emitHlslDecl(ctx: var PrintCtx, t: CirType, ident: string): string =
  ## HLSL declaration of `t` for `ident`, array dimensions composed into the name.
  if t.kind == ctArray:
    result = emitHlslScalar(ctx, t.aElem) & ' ' & ident & "[" & $t.aN & "]"
  else:
    result = emitHlslScalar(ctx, t) & ' ' & ident

proc emitBpName(t: CirType): string =
  ## Typedef name of the `vk::BufferPointer` alias for one buffer element type
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

proc bpElementStride(ctx: var PrintCtx, elem: CirType): int =
  ## Byte stride of one buffer element on the fvVulkan route, its storage width.
  ## The address arithmetic indexes buffers by it.
  let (stride, err) = bpElementStrideCheck(elem)
  if err.len > 0:
    ctx.fail(err)
  result = stride

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

proc emitCoordComponent(ctx: var PrintCtx, kind: CoordBuiltinKind, field: string): string =
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

proc emitCallArg(ctx: var PrintCtx, callee: CirExpr, i: int, a: CirExpr): string =
  ## One call argument: an address-of over a variable argument to a by-reference parameter
  ## emits the variable itself, the `inout` binding takes lvalues directly.
  if a.kind == eAddr and a.addrOperand.kind == eSym and
      callee.kind == eSym and ctx.procs.hasKey(callee.name):
    let fn = ctx.procs[callee.name]
    if i < fn.params.len and fn.params[i].typ.kind == ctPtr and
        fn.params[i].typ.space == caThread:
      return a.addrOperand.name
  return ctx.printExpr(a)

proc printCast(ctx: var PrintCtx, e: CirExpr): string =
  ## One bit reinterpret, the HLSL asint/asuint/asfloat reinterprets,
  ## 32-bit widths only.
  let t = e.castType
  let fn = if t.kind == ctInt and t.bits == 32: "asint"
    elif t.kind == ctUint and t.bits == 32: "asuint"
    elif t.kind == ctFloat and t.bits == 32: "asfloat"
    else: ""
  if fn.len == 0:
    ctx.fail("bit cast to " & emitHlslTypeBare(ctx, t) &
      " has no HLSL emission, only the 32-bit asint/asuint/asfloat reinterprets")
    return ""
  return fn & "(" & ctx.printExpr(e.castOperand) & ")"

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One numeric conversion, the constructor-call form.
  return emitHlslTypeBare(ctx, e.castType) & "(" & ctx.printExpr(e.castOperand) & ")"

proc printCallArg(ctx: var PrintCtx, callee: CirExpr, i: int, a: CirExpr): string =
  ## One call argument: a buffer resource passes by name, each call site
  ## supplies the resource bound to its own register, other arguments take
  ## the address-of policy of `emitCallArg`.
  if ctx.flavor == fvD3D12 and a.kind == eSym and
      ctx.bufferParams.contains(a.name):
    return a.name
  return ctx.emitCallArg(callee, i, a)


proc printBuiltinCall(ctx: var PrintCtx, call: CirExpr, args: seq[string]): string =
  ## Returns the HLSL text of an expression-position builtin call,
  ## "" when `call.callee` is not a builtin.
  if call.callee.kind == eSym:
    case call.callee.synchroBuiltin
    of sbkThreadgroupBarrier, sbkThreadgroupBarrierDevice:
      ctx.fail("barrier is a statement, a value-position barrier is outside the HLSL subset")
      return ""
    else:
      discard
    if call.callee.reductionBuiltin != rbkNone:
      ctx.fail("subgroup reductions have no emission, wave intrinsics are outside this backend")
      return ""
    if call.callee.atomicsBuiltin != abkNone:
      ctx.fail("atomic builtins have no emission in this backend")
      return ""
    case call.callee.shuffleBuiltin
    of shkSimdShuffle:
      # simd_shuffle maps to WaveReadLaneAt, a dynamically uniform
      # lane index is required by the intrinsic
      return "WaveReadLaneAt(" & args.join(", ") & ")"
    of shkSimdShuffleXor:
      ctx.fail("simd_shuffle_xor has no wave-intrinsic form, no HLSL intrinsic exchanges with a lane-index xor")
      return ""
    of shkNone: discard
    if call.callee.builtin and call.callee.name in hlslConvRow().fpRefusals:
      ctx.fail(call.callee.name & " has no HLSL form, the fp16/bf16/fp8" &
        " conversion builtins are outside the HLSL subset")
      return ""
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════


proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## HLSL text of the kinds in `targetExprKinds`.
  case e.kind
  of eNil:
    ctx.fail("nil literal is outside the HLSL subset")
    return ""
  of eSym:
    # coordinate builtins bind the kernel entry point, push constants qualify, buffer parameters take the flavor's route, names pass through
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
      if e.coordBuiltin == cbkThreadIndexInThreadgroup:
        ctx.coordReads.incl(e.coordBuiltin)
        return "int(" & coordInputName(e.coordBuiltin) & ")"
      ctx.fail("whole-vector read of a coordinate builtin is outside the" &
        " HLSL subset, read the components one by one")
      return ""
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
    # zero-descriptor route reads through the u64 device address, buffer parameters index by name, arrays index plainly
    let base = if e.kind == eAt: e.atBase else: e.patBase
    let idx = if e.kind == eAt: e.atIndex else: e.patIndex
    let idxTxt = ctx.printExpr(idx)
    if ctx.flavor == fvVulkan and base.kind == eSym and
        ctx.addressParams.hasKey(base.name):
      let elem = ctx.addressParams[base.name].pElem
      let stride = ctx.bpElementStride(elem)
      if ctx.failing():
        return ""
      return emitBpName(elem) & "(" & ctx.printExpr(base) &
        " + (uint64_t)(" & idxTxt & ") * " & $stride & ").Get()"
    if base.kind == eSym and ctx.bufferParams.contains(base.name):
      return base.name & "[" & idxTxt & "]"
    let bt = getExprType(base)
    if bt != nil and bt.kind == ctArray:
      var baseTxt = ctx.printExpr(base)
      if base.kind in {eCast, eConv}:
        # a cast binds looser than the subscript: parenthesize the base
        baseTxt = "(" & baseTxt & ")"
      return baseTxt & "[" & idxTxt & "]"
    ctx.fail("index of a " & (if bt == nil: "untyped" else: $bt.kind) &
      " base is outside the HLSL subset")
    return ""
  of eDot:
    # a coordinate component takes its semantic route, a vector component
    # takes its letter, names pass through
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
      return emitCoordComponent(ctx, e.dotBase.coordBuiltin, e.fieldName)
    let bt = getExprType(e.dotBase)
    if bt != nil and bt.kind == ctVector:
      let comp = emitVectorComponent(e.fieldName)
      if comp.len == 0:
        ctx.fail("no vector component " & e.fieldName)
        return ""
      return ctx.printExpr(e.dotBase) & "." & comp
    return ctx.printExpr(e.dotBase) & "." & e.fieldName
  of eDeref:
    # refused, HLSL has no pointer values
    ctx.fail("bare dereference has no emission, HLSL has no pointer values")
    return ""
  of eAddr:
    # refused, buffers are index-only resources and locals rewrite to their pointee
    ctx.fail("address-of has no HLSL form, buffers are index-only resources" &
      " and locals rewrite to their pointee")
    return ""
  of eCast:
    return ctx.printCast(e)
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    # refused, HLSL struct layout differs from the buffer layout the size would name
    ctx.fail("sizeof has no HLSL emission, HLSL struct layout differs from the" &
      " buffer layout the size would name")
    return ""
  of eOconstr:
    # vectors construct by call, valid in declaration initializers and assignments alike, where struct brace lists only initialize
    if e.ocType.kind == ctVector:
      var parts: seq[string]
      for kvv in e.kvFields:
        parts.add ctx.printExpr(kvv.value)
      return emitHlslScalar(ctx, e.ocType) & "(" & parts.join(", ") & ")"
    return ctx.printBraceList(e.ocType, e)
  of eAconstr:
    # one array construction in braces
    var parts: seq[string]
    for el in e.elems:
      parts.add ctx.printExpr(el)
    return "{ " & parts.join(", ") & " }"
  of eCall:
    var args: seq[string]
    for i, a in e.args:
      args.add ctx.printCallArg(e.callee, i, a)
    return printCallText(ctx, e, args, printBuiltinCall(ctx, e, args))
  else:
    # dispatcher sends only the kinds above, anything else is a routing bug
    ctx.fail("expression kind " & $e.kind & " has no HLSL emission")


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

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in print_common
# ═══════════════════════════════════════════════════════════════════════
proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: pointer and storage-class refusals, then the declaration text with its initializer.
  let pad = spaces(ctx.indent * 2)
  if s.declType.kind in {ctPtr, ctAptr}:
    ctx.fail("pointer local " & s.name & ", HLSL has no pointer values," &
      " the HLSL lowering pass rewrites pointer locals to their pointees")
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
  var decl = ctx.emitHlslDecl(s.declType, s.name)
  if s.init != nil:
    if s.init.kind == eOconstr and s.declType.kind == ctStruct:
      decl &= " = " & ctx.printBraceList(s.declType, s.init)
    else:
      decl &= " = " & ctx.printExpr(s.init)
  result = pad & decl & ";"

proc printConst(ctx: var PrintCtx, s: CirStmt): string =
  ## One local constant declaration, struct initializers print as a brace list.
  let pad = spaces(ctx.indent * 2)
  if s.init != nil and s.init.kind == eOconstr and s.declType.kind == ctStruct:
    result = pad & "const " & ctx.emitHlslDecl(s.declType, s.name) &
      " = " & ctx.printBraceList(s.declType, s.init) & ";"
  else:
    result = pad & "const " & ctx.emitHlslDecl(s.declType, s.name) &
      " = " & ctx.printExpr(s.init) & ";"

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment: buffer, push-constant, struct and array-constructor policy, then the assignment text.
  let pad = spaces(ctx.indent * 2)
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

proc printCall(ctx: var PrintCtx, s: CirStmt): string =
  ## One call statement: the buffer-argument policy and barrier intrinsics, then the plain call form.
  let pad = spaces(ctx.indent * 2)
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
      parts.add ctx.emitCallArg(s.callCallee, i, a)
  if s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrier:
    result = pad & "GroupMemoryBarrierWithGroupSync();"
  elif s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
    result = pad & "DeviceMemoryBarrierWithGroupSync();"
  else:
    result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: refused, the text names the target.
  ctx.fail("inline assembly has no HLSL form")
  result = ""

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitResourceParam(ctx: var PrintCtx, p: CirParam, rw: bool): string =
  ## HLSL resource parameter of a buffer: StructuredBuffer / RWStructuredBuffer
  ## over the element type, `rw` picks the read-write form.
  let resTy = if rw: "RWStructuredBuffer" else: "StructuredBuffer"
  result = resTy & "<" & ctx.emitHlslScalar(p.typ.pElem) & "> " & p.name

proc emitFnSignature(ctx: var PrintCtx, fn: CirProc, graph: CallGraph, forwardDeclare: bool): string =
  ## `RetType name(params)` of one proc, buffers as resources, `var T` device parameters as `inout`.
  ## The signature of an `{.inline.}` proc opens with the `inline ` qualifier.
  ## Parameter legality refuses in the classification pass, the printer reads the stored records.
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      if ctx.flavor == fvVulkan:
        # address value only: reads and writes construct the buffer pointer per access
        params.add "uint64_t " & p.name
      else:
        let rw = graph.reached.contains(fn.name & "/" & p.name)
        params.add ctx.emitResourceParam(p, rw)
    of ctPtr:
      params.add "inout " & ctx.emitHlslDecl(p.typ.pElem, p.name)
    else:
      params.add ctx.emitHlslDecl(p.typ, p.name)
  if ctx.failing():
    return ""
  let ret = ctx.emitHlslScalar(fn.ret)
  if ctx.failing():
    return ""
  if fn.inline:
    result = "inline "
  result &= ret & ' ' & fn.name & '(' & params.join(", ") & ')'
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
  let bodyTxt = ctx.printStmts(fn.body)
  if ctx.failing():
    return ""
  let sig = ctx.emitFnSignature(fn, graph, forwardDeclare = false)
  if ctx.failing():
    return ""
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc emitStructDecls(ctx: var PrintCtx, refs: HashSet[string],
    byName: Table[string, CirType], emitted: var HashSet[string],
    name: string, outp: var string) =
  ## Emits one struct definition, member structs first, deduplicated by name.
  ##
  ## - HLSL resolves types at their declaration site, a referenced struct
  ##   holding a pointer refuses, HLSL structs cannot take resources
  ## - the reference set is the stored record `CirModule.structRefs`, every
  ##   struct the member recursion reaches is already in it
  if name in emitted or name notin refs:
    return
  let t = byName.getOrDefault(name, nil)
  if t == nil:
    ctx.fail("struct " & name & " is referenced but has no declaration")
    return
  if t.containsPtrDeep:
    ctx.fail("struct " & name & " holds a pointer, HLSL structs cannot hold" &
      " resources, the HLSL lowering pass rewrites pointer-holding locals to their fields")
    return
  emitted.incl name
  for f in t.fields:
    if f.typ.kind == ctStruct:
      ctx.emitStructDecls(refs, byName, emitted, f.typ.name, outp)
    elif f.typ.kind == ctArray:
      if f.typ.aElem.kind == ctStruct:
        ctx.emitStructDecls(refs, byName, emitted, f.typ.aElem.name, outp)
  if ctx.failing():
    return
  outp.add "struct " & name & " {\n"
  if t.fields.len == 0:
    # HLSL requires at least one member in a struct
    outp.add "  uint _padding;\n"
  else:
    for f in t.fields:
      outp.add "  " & ctx.emitHlslDecl(f.typ, f.name) & ";\n"
  outp.add "};\n\n"

proc emitMainInputs(ctx: PrintCtx): string =
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
  ## then the kernel entry: `main`, or the `{.exportc: "name".}` override.
  ##
  ## - fvD3D12: register resources and the `ConstantBuffer` push block
  ## - fvVulkan: the `[[vk::push_constant]]` struct with every member offset locked
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>", flavor: flavor,
    policy: hlslKindRow(), lit: litTokens(hlslLitRow(), hlslScalarRow()))
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

  # read-write classification of the buffer parameters: the stored record
  # computed by the call-graph analysis pass after the last transform
  let graph = m.callGraph

  # kernel resources: buffer parameters and Push members, read from the stored `CirModule.paramClasses` record the classification pass filled
  ctx.fnName = kernel.name
  let kernelClass = m.paramClasses[kernel.name]
  let pushParams = kernelClass.push
  let buffers = kernelClass.buffers

  # struct declarations for the referenced value structs, dependency order, read from the stored `CirModule.structRefs` record
  let refs = m.structRefs
  var byName: Table[string, CirType]
  for t in m.types:
    if t.kind == ctStruct:
      byName[t.name] = t
  var emitted = initHashSet[string]()
  var refList: seq[string]
  for name in refs:
    refList.add name
  for name in refList:
    ctx.emitStructDecls(refs, byName, emitted, name, result.src)
  if ctx.failing():
    return ("", ctx.err)

  # binding section: fvD3D12 register resources, fvVulkan push-constant struct with locked offsets
  if ctx.flavor == fvVulkan:
    # buffer-pointer aliases, one per element type in encounter order
    var bpEmitted: HashSet[string]
    for p in buffers:
      let nm = emitBpName(p.typ.pElem)
      if nm notin bpEmitted:
        bpEmitted.incl nm
        result.src &= "typedef vk::BufferPointer<" &
          ctx.emitHlslScalar(p.typ.pElem) & "> " & nm & ";\n"
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
            ctx.emitHlslDecl(p.typ, p.name) & ";\n"
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
      result.src &= ctx.emitResourceParam(p, rw) & " : register(" & reg &
        $ordinal & ", space0);\n"
      ordinal.inc
    if pushParams.len > 0:
      result.src &= "struct Push {\n"
      for p in kernel.params:
        if p.name == "crucible_numworkgroups":
          # materialized dispatch grid size, an int3 the engine stages
          result.src &= "  int3 " & p.name & ";\n"
        elif p.name in pushParams:
          result.src &= "  " & ctx.emitHlslDecl(p.typ, p.name) & ";\n"
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
        hoisted.add "static const " & ctx.emitHlslDecl(s.declType, s.name) &
          " = " & ctx.printBraceList(s.declType, s.init) & ";"
      else:
        hoisted.add "static const " & ctx.emitHlslDecl(s.declType, s.name) &
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
    result.src &= ctx.emitFnSignature(fn, graph, forwardDeclare = true) & '\n'
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
  let bodyTxt = ctx.printStmts(kernel.body)
  if ctx.failing():
    return ("", ctx.err)
  result.src &= "[numthreads(" & $ctx.workgroup[0] & ", " & $ctx.workgroup[1] &
    ", " & $ctx.workgroup[2] & ")]\n"
  let inputs = ctx.emitMainInputs()
  # exportc renames a kernel entry for the host runtime lookup. Vulkan ingest reads
  # back the entry name from the emitted `[numthreads]` line and passes it to dxc.
  # Device functions keep the name their call sites reference, a bare kernel stays `main`.
  let entryName = if kernel.exportName.len > 0: kernel.exportName else: "main"
  result.src &= "void " & entryName & "(" & inputs & ")\n{\n" & bodyTxt & "\n}\n"
  return (result.src, "")

proc codegen*(ctx: var GpuContext, flavor: HlslFlavor = fvD3D12): string =
  ## Prints the module in `ctx` as HLSL source for `flavor`, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printHlslModule(ctx.module, flavor))
