## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
import std / [sequtils, tables]
import ../ir/gpu_types
import ./pass_datatypes


proc isLvalue*(n: GpuAst): bool =
  ## Returns true if the AST node is an lvalue (can have its address taken).
  case n.kind
  of gpuIdent: true
  of gpuIndex: true
  of gpuDeref: true
  else: false

proc walkNonLvalueArgs(ctx: var GpuContext; n: var GpuAst) =
  case n.kind
  of gpuCall:
    let fnParams = ctx.getFnParams(n.cName)
    for i, arg in n.cArgs:
      if i < fnParams.len and fnParams[i].passByRef and not arg.isLvalue():
        n.cArgs[i] = GpuAst(kind: gpuMaterialize,
          mExpr: arg,
          mType: fnParams[i].typ)
    for ch in n.mitems:
      walkNonLvalueArgs(ctx, ch)
  else:
    for ch in n.mitems:
      walkNonLvalueArgs(ctx, ch)

proc materializePassByRefArgs*(ctx: var GpuContext) =
  ## Transforms non-lvalue arguments to passByRef parameters into
  ## gpuMaterialize nodes that backends can handle appropriately.
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    walkNonLvalueArgs(ctx, fn.pBody)


# ═══════════════════════════════════════════════════════════════════
# Fold max/min idioms to backend-native builtins
# ═══════════════════════════════════════════════════════════════════

const
  BasicNumericKinds* = {gtUint8, gtUint16, gtInt16, gtUint32, gtInt32,
                        gtUint64, gtInt64, gtFloat32, gtFloat64, gtFloat16,
                        gtBf16, gtSize_t}
    ## Types the backends' native max/min accept. Int[N] structs (gtObject)
    ## are deliberately excluded — ceramic's genBinOp handles those.

proc isBasicNumeric*(t: GpuType): bool =
  not t.isNil and t.kind in BasicNumericKinds

proc operandType*(ctx: GpuContext; n: GpuAst): GpuType =
  ## Best-effort type of a pattern operand. Returns nil when the pass
  ## cannot determine it — the fold then conservatively skips.
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.operandType(n.pVal)  # `-x` keeps x's type
  of gpuCall: ctx.getFnReturnType(n.cName)
  else: nil

proc sameExpr(a, b: GpuAst): bool =
  ## Structural equality for pattern operands. iSym is the immutable
  ## fingerprint (name may be mangled). NOTE: GpuAst has a custom `==`
  ## (structural, raiseAssert on non-idents) — never use `==`/`!=` with
  ## nil here; use isNil.
  if a.isNil or b.isNil: return false
  if a.kind != b.kind: return false
  case a.kind
  of gpuIdent:
    result = a.symbol.iSym == b.symbol.iSym or
             a.symbol.name == b.symbol.name
  of gpuLit:
    result = a.lValue == b.lValue and a.lType.kind == b.lType.kind
  of gpuBinOp:
    result = a.bOp.symbol.name == b.bOp.symbol.name and
             sameExpr(a.bLeft, b.bLeft) and sameExpr(a.bRight, b.bRight)
  of gpuPrefix:
    result = a.pOp == b.pOp and sameExpr(a.pVal, b.pVal)
  of gpuCall:
    result = a.cName.symbol.name == b.cName.symbol.name and
             a.cArgs.len == b.cArgs.len and
             a.cArgs.zip(b.cArgs).allIt(sameExpr(it[0], it[1]))
  else: discard

proc foldMinMaxPattern(ctx: GpuContext; n: var GpuAst): bool =
  ## If `n` is a gpuTernary matching the max/min idiom, replaces it with a
  ## call to the backend-native max/min builtin (plain name — the
  ## ambiguous-builtin name every backend supports for basic types).
  ##
  ## Matched shapes (operands A = tThen, B = tElse):
  ##   max: (B <= A) ? A : B            — generic [T: not SomeFloat] body
  ##        ((B <= A) || !(B == B)) ? A : B  — float32/64 body (NaN guard)
  ##   min: (A <= B) ? A : B            — generic [T: not SomeFloat] body
  ##        ((A <= B) || !(B == B)) ? A : B  — float32/64 body (NaN guard)
  ##
  ## KNOWN semantic trade-off (documented): the ternary is Nim's exact
  ## semantics; the builtins differ on signed zeros — IEEE fmax returns
  ## +0.0 for fmax(+0.0, -0.0), Nim's guard form returns -0.0. Accepted;
  ## the hardware instruction is the point of the pass.
  if n.isNil: return false
  if n.kind != gpuTernary: return false
  if n.tCond.isNil: return false
  if n.tThen.isNil or n.tElse.isNil: return false
  let tThen = n.tThen
  let tElse = n.tElse

  # Condition: Le(P, Q), optionally OR'd with a NaN guard !(X == X).
  var le: GpuAst = nil
  var hasGuard = false
  proc isLe(x: GpuAst): bool =
    not x.isNil and x.kind == gpuBinOp and not x.bOp.isNil and
    x.bOp.kind == gpuIdent and not x.bOp.symbol.isNil and
    x.bOp.symbol.name == "<="
  proc isNanGuard(x: GpuAst): bool =
    not x.isNil and x.kind == gpuPrefix and x.pOp == "!" and not x.pVal.isNil and
    x.pVal.kind == gpuBinOp and not x.pVal.bOp.isNil and
    x.pVal.bOp.kind == gpuIdent and not x.pVal.bOp.symbol.isNil and
    x.pVal.bOp.symbol.name == "==" and
    sameExpr(x.pVal.bLeft, x.pVal.bRight)
  case n.tCond.kind
  of gpuBinOp:
    if not n.tCond.bOp.isNil and n.tCond.bOp.kind == gpuIdent and
       not n.tCond.bOp.symbol.isNil and n.tCond.bOp.symbol.name == "<=":
      le = n.tCond
    elif not n.tCond.bOp.isNil and n.tCond.bOp.kind == gpuIdent and
         not n.tCond.bOp.symbol.isNil and n.tCond.bOp.symbol.name == "||":
      if isLe(n.tCond.bLeft) and isNanGuard(n.tCond.bRight):
        le = n.tCond.bLeft; hasGuard = true
      elif isNanGuard(n.tCond.bLeft) and isLe(n.tCond.bRight):
        le = n.tCond.bRight; hasGuard = true
  else: discard
  if le.isNil: return false
  if le.bLeft.isNil or le.bRight.isNil: return false

  let P = le.bLeft
  let Q = le.bRight
  var builtin = ""
  if sameExpr(tElse, P) and sameExpr(tThen, Q):
    builtin = "max"
  elif sameExpr(tThen, P) and sameExpr(tElse, Q):
    builtin = "min"
  else:
    return false

  # Type gate: both operands basic numerics of the same type.
  let tA = ctx.operandType(tThen)
  let tB = ctx.operandType(tElse)
  if not isBasicNumeric(tA) or not isBasicNumeric(tB): return false
  if tA.kind != tB.kind: return false
  # The unguarded form on floats is NOT fmax: NaN propagates to tElse
  # ((y <= x) with y NaN is false → returns y). Only the NaN-guarded
  # form is fmax-equivalent on floats.
  if not hasGuard and tA.kind in {gtFloat32, gtFloat64, gtFloat16, gtBf16}: return false

  var callName = GpuAst(kind: gpuIdent, symbol: newSymbol(builtin))
  n = GpuAst(kind: gpuCall, cIsExpr: true, cName: callName,
             cArgs: @[tThen, tElse])
  return true

proc foldMaxMinInBody(ctx: var GpuContext; n: var GpuAst) =
  ## Recursive pre-order walker (walk's closure cannot capture the
  ## GpuContext value type). foldMinMaxPattern may replace `n` with a gpuCall
  ## whose args are the same operand refs — recursing into the children
  ## still visits the original operand expressions.
  discard ctx.foldMinMaxPattern(n)
  for ch in n.mitems:
    ctx.foldMaxMinInBody(ch)

proc foldMaxMinToBuiltins*(ctx: var GpuContext) =
  ## Rewrites ternary max/min idioms to the backend-native max/min
  ## builtins so they lower to the hardware instruction (e.g. PTX
  ## max.f32 / max.u32 / max.s32) instead of a compare+select sequence.
  ##
  ## The fold is a pure expression rewrite: it replaces the ternary
  ## SUBTREE in place, so both direct (`result = ternary`) and blitted
  ## (`_blit_N = ternary; result = _blit_N`) forms are handled. The
  ## rewritten device functions stay contained; backends inline them.
  ##
  ## TODO(abs): fold `(x < 0) ? -x : x` to the backend abs builtin too.
  ## Blocked on two things: (1) Nim's float abs body contains `when nimvm:`
  ## which crashes translation before any pass sees it; (2) float abs needs
  ## a PER-BACKEND name table — CUDA fabsf/fabs, OpenCL fabs (abs is
  ## int-only there), GLSL/Vulkan abs, WGSL abs. Int abs is uniform (abs)
  ## and already native (magic AbsI), so the table is floats-only.
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    ctx.foldMaxMinInBody(fn.pBody)

proc registerOptimizationPasses*(reg: var PassRegistry) =
  ## Register optimization passes. Runs after preprocessing (mangleNames)
  ## so constructed builtin calls keep their plain name.
  reg.register("foldMaxMinToBuiltins", pkTransform, phaseMain,
    "Folds ternary max/min idioms to backend-native max/min builtins",
    dependsOn = @["mangleNames"],
    run = foldMaxMinToBuiltins
  )