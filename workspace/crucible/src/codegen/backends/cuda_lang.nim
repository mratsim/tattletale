# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CUDA C++ printer: a structured CIRv2 module to NVRTC-compilable CUDA C++ source,
## table-driven over the closed type lattice, structured like the MSL printer.
##
## Signature and storage contract:
##
## - kernels: `extern "C" __global__ void name(args)`, the host runtime
##   looks up the kernel by unmangled name
## - device functions: `__device__` with no extern "C", each device function keeps its C++ mangled name.
## - parameters print plainly, CUDA has one generic address space,
##   `{.smem.}` locals print `__shared__` in the body, and program-scope variables in `CirModule.globals` print `__constant__`
##
## Statement and builtin contract:
##
## - coordinate builtins map to the CUDA vocabulary: blockIdx, blockDim,
##   gridDim, threadIdx. The canonical Nim type is int, the CUDA builtins are unsigned, reads cast to int
## - `{.unroll.}` prints `#pragma unroll` in front of the loop, jmp to the exit
##   label is `break`, jmp to the continue label is `continue`, any other lab/jmp pair refuses
## - the typed builtin kinds dispatch to the CUDA vocabulary, `{.builtin.}` names pass through, the warp reductions are the exception
## - a warp-reduction call makes its butterfly template emit at module scope on first use

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables, math]
import ../ir/ast_datatypes
import ./emit_common

type
  PrintError* = object
    ## Refusal of the printer, names the proc and the cause.
    procName*: string
    msg*: string

  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    ## Per-kernel state.
    builtins: OrderedSet[string]
    ## Canonical builtin names that the kernel body references, first-use order.

const printerTarget = "cuda"

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc spellCudaScalar(t: CirType): string =
  ## CUDA name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid:
    result = "void"
  of ctBool:
    result = "bool"
  of ctInt:
    case t.bits
    of 8:
      result = "signed char"
    of 16:
      result = "short"
    of 32:
      result = "int"
    of 64:
      result = "long long"
    else:
      result = ""
  of ctUint:
    case t.bits
    of 8:
      result = "unsigned char"
    of 16:
      result = "unsigned short"
    of 32:
      result = "unsigned int"
    of 64:
      result = "unsigned long long"
    else:
      result = ""
  of ctFloat:
    case t.bits
    of 32:
      result = "float"
    of 64:
      result = "double"
    else:
      result = ""
  of ctVector:
    # CUDA vector types, the short scalar names that the CUDA headers define
    let elem = case t.vElem.kind
      of ctInt:
        case t.vElem.bits
        of 8: "char"
        of 16: "short"
        of 32: "int"
        of 64: "longlong"
        else: ""
      of ctUint:
        case t.vElem.bits
        of 8: "uchar"
        of 16: "ushort"
        of 32: "uint"
        of 64: "ulonglong"
        else: ""
      of ctFloat:
        case t.vElem.bits
        of 32: "float"
        of 64: "double"
        else: ""
      else: ""
    if elem.len > 0:
      result = elem & $t.vN
  of ctPtr, ctAptr:
    result = spellCudaScalar(t.pElem) & "*"
  of ctStruct:
    result = t.name
  of ctArray, ctProc:
    result = ""

proc spellCudaType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## CUDA name of `t` with `ident` appended. Arrays take the identifier into
  ## the declarator (`T ident[N]`, pointer-to-array `T (*ident)[N]`).
  ## `allowEmptyIdent` permits no identifier, return types and casts use it.
  ## An array without an identifier otherwise refuses.
  case t.kind
  of ctArray:
    if ident.len == 0 and not allowEmptyIdent:
      return "<cuda: array needs an identifier>"
    if t.aElem.kind == ctArray:
      result = spellCudaType(t.aElem, "(" & ident & ")" & "[" & $t.aN & "]",
        allowEmptyIdent)
    else:
      result = spellCudaType(t.aElem, "", true) & ' ' & ident & '[' & $t.aN & ']'
  else:
    result = spellCudaScalar(t)
    if ident.len > 0:
      result.add ' ' & ident

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(ctx: var PrintCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc spellCudaLit(e: CirExpr): string =
  ## CUDA token of a literal, typed by its literal type.
  case e.kind
  of eIntLit:
    case e.litType.kind
    of ctUint:
      case e.litType.bits
      of 64: result = $e.value & "ULL"
      of 32: result = $e.value & "U"
      else: result = "(" & spellCudaScalar(e.litType) & ")" & $e.value
    of ctInt:
      case e.litType.bits
      of 64: result = $e.value & "LL"
      of 32: result = $e.value
      else: result = "(" & spellCudaScalar(e.litType) & ")" & $e.value
    else:
      result = $e.value
  of eFloatLit:
    # CUDA has no `inf`/`nan` tokens and Nim's `$` renders `inf`/`nan`,
    # emission text is the <math.h> macros that the CUDA compiler accepts
    case e.fValue.classify
    of fcInf:
      result = "INFINITY"
    of fcNegInf:
      result = "-INFINITY"
    of fcNan:
      result = "NAN"
    else:
      case e.fLitType.bits
      of 32: result = $e.fValue & "f"
      else: result = $e.fValue
  of eNil: result = "nullptr"
  of eTrue: result = "true"
  of eFalse: result = "false"
  else: result = "<cuda: not a literal>"

proc cudaCoordToken(kind: CoordBuiltinKind, name: string, whole: bool, field: string, coordTy: CirType): string =
  ## CUDA text of a canonical coordinate reference.
  ## - whole-value use of a vector builtin: a braced init of the coordinate
  ##   struct the module registered. The CUDA builtins are unsigned and
  ##   the canonical Nim type is int, the components cast
  ## - component use: a cast to int, the flat thread index linearizes
  ##   x-major and parenthesizes so a trailing `* k` cannot mis-associate
  let vecTy = spellCudaScalar(coordTy)
  case kind
  of cbkThreadgroupPositionInGrid:
    if whole:
      result = vecTy & "{(int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z}"
    else:
      result = "(int)(blockIdx." & field & ")"
  of cbkThreadPositionInThreadgroup:
    if whole:
      result = vecTy & "{(int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z}"
    else:
      result = "(int)(threadIdx." & field & ")"
  of cbkThreadsPerThreadgroup:
    if whole:
      result = vecTy & "{(int)blockDim.x, (int)blockDim.y, (int)blockDim.z}"
    else:
      result = "(int)(blockDim." & field & ")"
  of cbkThreadgroupsPerGrid:
    if whole:
      result = vecTy & "{(int)gridDim.x, (int)gridDim.y, (int)gridDim.z}"
    else:
      result = "(int)(gridDim." & field & ")"
  of cbkThreadIndexInThreadgroup:
    if whole:
      result = "(int)(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)"
    else:
      result = "(int)(threadIdx." & field & ")"
  of cbkThreadPositionInGrid:
    # whole-value use has no CUDA text, the canonical name passes through
    if whole:
      result = name
    else:
      result = "(int)(blockIdx." & field & "*blockDim." & field &
        "+threadIdx." & field & ")"
  of cbkNone:
    result = name

proc spellCudaBuiltinCall(callee: CirExpr, args: seq[string]): string =
  ## Returns the CUDA text of a builtin call, "" when `callee` is not a builtin.
  ##
  ## Shared by expression- and statement-position calls: the statement path
  ## bypasses `printExpr`, so a barrier printed as a bare name would lose
  ## its memory ordering, the CUDA text is `__syncthreads()`.
  ##
  ## `args` contains the already-printed argument texts.
  if callee.kind != eSym:
    return ""
  case callee.synchroBuiltin
  of sbkThreadgroupBarrier, sbkThreadgroupBarrierDevice:
    # __syncthreads() already orders global memory, the device-scope barrier needs no separate instruction
    return "__syncthreads()"
  of sbkNone: discard
  case callee.reductionBuiltin
  of rbkReduceAdd:
    return "warpReduceAdd(" & args.join(", ") & ")"
  of rbkReduceMax:
    return "warpReduceMax(" & args.join(", ") & ")"
  of rbkNone: discard
  if callee.atomicsBuiltin == abkAtomicFetchAdd:
    # atomicAdd overloads on the pointee type, no cast needed
    return "atomicAdd(" & args[0] & ", " & args[1] & ")"
  if callee.builtin and callee.name in ["simd_shuffle", "simd_shuffle_xor"]:
    # full mask: every warp lane participates and receives the result,
    # xor exchanges lane pairs bitwise
    let shfl = if callee.name == "simd_shuffle": "__shfl_sync"
               else: "__shfl_xor_sync"
    return shfl & "(0xffffffffu, " & args[0] & ", " & args[1] & ")"
  return ""

proc printExpr(ctx: var PrintCtx, e: CirExpr): string =
  if ctx.failing():
    return ""
  case e.kind
  of eNil, eTrue, eFalse, eIntLit, eFloatLit:
    return spellCudaLit(e)
  of eSym:
    if e.coordBuiltin != cbkNone:
      ctx.builtins.incl(e.name)
      return cudaCoordToken(e.coordBuiltin, e.name, true, "", e.typ)
    return e.name
  of eAt:
    var base = ctx.printExpr(e.atBase)
    if e.atBase.kind in {eCast, eConv}:
      # a cast binds looser than the subscript: parenthesize the base
      base = "(" & base & ")"
    return base & '[' & ctx.printExpr(e.atIndex) & ']'
  of ePat:
    var base = ctx.printExpr(e.patBase)
    if e.patBase.kind in {eCast, eConv}:
      base = "(" & base & ")"
    return base & '[' & ctx.printExpr(e.patIndex) & ']'
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      ctx.builtins.incl(e.dotBase.name)
      return cudaCoordToken(e.dotBase.coordBuiltin, e.dotBase.name, false,
        e.fieldName, e.dotBase.typ)
    let base = ctx.printExpr(e.dotBase)
    if e.dotBase.kind == eDeref:
      return "(" & base & ")." & e.fieldName
    return base & '.' & e.fieldName
  of eDeref:
    return "(*" & ctx.printExpr(e.derefOperand) & ")"
  of eAddr:
    return "(&" & ctx.printExpr(e.addrOperand) & ")"
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
  of eConv, eCast:
    # a conversion or bit reinterpret prints as the C-style cast, CUDA has no as_type and one C cast covers both semantics here
    let t = e.castType
    return "(" & spellCudaType(t, "", true) & ")" & ctx.printExpr(e.castOperand)
  of eSizeof:
    return "sizeof(" & spellCudaType(e.ofType, "", true) & ")"
  of eOconstr:
    # Braced init list: `TypeName{val, ...}`, the functional-style cast form
    # keeps the result a valid C++ expression, bare braced-init-lists are
    # not expressions and cannot be used with member access
    result = spellCudaType(e.ocType, "", true) & "{"
    for i, kv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kv.value)
    result.add "}"
  of eAconstr:
    result = "{"
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add "(" & spellCudaType(e.acType.aElem, "", true) & ")" &
        ctx.printExpr(el)
    result.add "}"
  of eCall:
    let calleeName =
      if e.callee.kind == eSym: e.callee.name
      else: ctx.printExpr(e.callee)
    var args: seq[string]
    for a in e.args:
      args.add ctx.printExpr(a)
    let builtinTxt = spellCudaBuiltinCall(e.callee, args)
    if builtinTxt.len > 0:
      return builtinTxt
    return calleeName & "(" & args.join(", ") & ")"
  else:
    ctx.fail("expression kind " & $e.kind & " has no CUDA emission")
    return ""

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string
  ## Forward declaration, the loop context (break/continue labels) is a parameter.

proc printStmt(ctx: var PrintCtx, s: CirStmt, exitL, contL: string): string =
  ## One structured statement, `exitL`/`contL` the enclosing loop's labels.
  if ctx.failing():
    return ""
  let pad = spaces(ctx.indent * 2)
  case s.kind
  of sVar:
    if s.space == caThreadgroup:
      # kernel-scope shared storage in `__shared__` form,
      # NVRTC allocates it per threadgroup, any initializer is discarded
      result = pad & "__shared__ " & spellCudaType(s.declType, s.name, false) & ";"
      return
    if s.space == caConstant:
      # program-scope storage belongs to CirModule.globals, a survivor here
      # was never relocated
      ctx.fail("local " & s.name & " in constant space, CUDA constant storage" &
        " is program scope only")
      return ""
    var decl = spellCudaType(s.declType, s.name, false)
    result = pad & decl
    if s.init != nil:
      result &= " = " & ctx.printExpr(s.init)
    result &= ";"
  of sConst:
    if s.declType.kind == ctArray:
      result = pad & "constexpr " &
        spellCudaType(s.declType, s.name, false) & " = " & ctx.printExpr(s.init) & ";"
    else:
      result = pad & "constexpr " & spellCudaScalar(s.declType) & ' ' & s.name &
        " = " & ctx.printExpr(s.init) & ";"
  of sAsgn:
    result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"
  of sIte:
    let cond = ctx.printExpr(s.cond)
    ctx.indent.inc
    let thenTxt = ctx.printStmts(s.thenBody, exitL, contL)
    ctx.indent.dec
    result = pad & "if (" & cond & ") {\n" & thenTxt & '\n' & pad & '}'
    if s.elseBody.len > 0:
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
    # a loop-bound hint, CUDA has no assume construct
    result = ""
  of sRet:
    if s.retVal != nil:
      result = pad & "return " & ctx.printExpr(s.retVal) & ";"
    else:
      result = pad & "return;"
  of sDiscard:
    result = ""
  of sCall:
    let calleeName =
      if s.callCallee.kind == eSym: s.callCallee.name
      else: ctx.printExpr(s.callCallee)
    var args: seq[string]
    for a in s.callArgs:
      args.add ctx.printExpr(a)
    let builtinTxt = spellCudaBuiltinCall(s.callCallee, args)
    if builtinTxt.len > 0:
      result = pad & builtinTxt & ";"
    else:
      result = pad & calleeName & "(" & args.join(", ") & ");"
  of sAsm:
    var text = ""
    for i, part in s.asmParts:
      text.add(part)
      if i < s.asmOperands.len:
        text.add(s.asmOperands[i].name)
    result = pad & "asm(" & text.strip() & ");"
  of sEmit:
    # `{.emit.}` raw source: the operand names splice into the verbatim text,
    # no terminator appended, the text supplies its own terminators
    var text = ""
    for i, part in s.emitParts:
      text.add(part)
      if i < s.emitOperands.len:
        text.add(s.emitOperands[i].name)
    result = pad & text
  of sCase:
    ctx.fail("case statements are outside the CIRv2 closure, the frontend refuses them")

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string =
  ## Prints one statement list. Plain statements take the caller's loop context.
  ## A `loop` derives its own labels. Its exit label sits after the loop,
  ## its continue label inside.
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
      let directive = if s.unroll:
        spaces(ctx.indent * 2) & "#pragma unroll\n"
      else:
        ""
      txt = directive & spaces(ctx.indent * 2) & "for (;;) {\n" & bodyTxt & "\n" &
        spaces(ctx.indent * 2) & '}'
      if lExit.len > 0:
        i.inc  # exit label, consumed
    else:
      txt = ctx.printStmt(s, exitL, contL)
    if txt.len > 0:
      lines.add txt
    i.inc
  return lines.join("\n")

proc spellWarpReduceTemplate(kind: ReductionBuiltinKind): string =
  ## Butterfly warp-reduction template for one reduction kind: CUDA has no
  ## float warp-reduce intrinsic, the tree walks the xor mask 16/8/4/2/1.
  let op = case kind
    of rbkReduceAdd: "v += "
    else: "v = max(v, "
  let close = case kind
    of rbkReduceAdd: ""
    else: ")"
  result = "template <typename T> __forceinline__ __device__ T warpReduce" &
    (if kind == rbkReduceAdd: "Add" else: "Max") & "(T v) {\n"
  result &= "  #pragma unroll\n"
  result &= "  for (int delta = 16; delta > 0; delta >>= 1) {\n"
  result &= "    " & op & "__shfl_xor_sync(0xffffffff, v, delta, 32)" & close & ";\n"
  result &= "  }\n"
  result &= "  return v;\n"
  result &= "}\n\n"

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc printProc(ctx: var PrintCtx, fn: CirProc): string =
  ## One proc or kernel: signature, body, closing brace. The body prints first,
  ## the kernel signature consumes the builtin set the body references.
  ctx.fnName = fn.name
  ctx.builtins = initOrderedSet[string]()
  ctx.indent = 0
  let bodyTxt = ctx.printStmts(fn.body, "", "")
  if ctx.failing():
    return ""
  let name = if fn.cudaName.len > 0: fn.cudaName else: fn.name
  var params: seq[string]
  if fn.isKernel:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        params.add spellCudaScalar(p.typ.pElem) & "* " & p.name
      of ctArray:
        # by-value array params decay to pointers, they are read-only,
        # const-qualified so the read-only contract reaches the callee
        params.add "const " & spellCudaType(p.typ, p.name, false)
      else:
        params.add spellCudaType(p.typ, p.name, true)
    for b in ctx.builtins:
      # coordinate builtins have no CUDA parameter, the vocabulary is global
      discard b
  else:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        params.add spellCudaScalar(p.typ.pElem) & "* " & p.name
      else:
        params.add spellCudaType(p.typ, p.name, true)
  let ret = spellCudaType(fn.ret, "", true)
  var prefix = ""
  if fn.isKernel:
    # extern "C" keeps the unmangled name that the host runtime looks up
    prefix = "extern \"C\" __global__ "
  else:
    if fn.forceInline:
      prefix = "__forceinline__ "
    prefix &= "__device__ "
  let sig = prefix & ret & ' ' & name & '(' & params.join(", ") & ")"
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc printCudaModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full CUDA translation unit: struct declarations, program-scope constants,
  ## the warp-reduction templates in use, then every function body.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>")
  var src = ""
  for t in m.types:
    if t.kind != ctStruct:
      ctx.fail("module type " & t.name & " is not a struct")
      break
    src &= "struct " & t.name & " {\n"
    if t.fields.len == 0:
      # CUDA requires at least one field in a struct
      src &= "  char _;\n"
    for f in t.fields:
      src &= "  " & spellCudaType(f.typ, f.name, false) & ";\n"
    src &= "};\n\n"
  for s in m.globals:
    if s.kind != sVar:
      ctx.fail("module global " & s.name & " is not a variable")
      break
    var decl = spellCudaType(s.declType, s.name, false)
    if s.init != nil:
      decl &= " = " & ctx.printExpr(s.init)
    src &= "__constant__ " & decl & ";\n"
  if m.globals.len > 0:
    src &= "\n"
  # warp-reduction template set from the stored module feature scan
  # (`CirModule.features`, filled by the feature-scan pass)
  let reduceKinds = m.features.reductions
  if rbkReduceAdd in reduceKinds:
    src &= spellWarpReduceTemplate(rbkReduceAdd)
  if rbkReduceMax in reduceKinds:
    src &= spellWarpReduceTemplate(rbkReduceMax)
  if ctx.failing():
    return ("", ctx.err)
  for fn in m.procs:
    let txt = ctx.printProc(fn)
    if ctx.err.len > 0:
      break
    src &= txt & "\n\n"
  result = (src, ctx.err)

proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as CUDA C++ source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printCudaModule(ctx.module))
