# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
import std / [sequtils, sets, tables]
import ../ir/gpu_ast_datatypes
import ../frontend/nim_to_gpu_type_resolvers
import ../ir/gpu_ast_type_resolvers
import ./pass_datatypes

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════

proc registerOptimizationPasses*(reg: var PassRegistry) =
  ## After mangleNames, so constructed builtin calls keep their plain name
  reg.register("foldMaxMinToBuiltins", pkTransform, phaseMain,
    "Folds ternary max/min idioms to backend-native max/min builtins",
    dependsOn = @["mangleNames"],
    run = foldMaxMinToBuiltins
  )
  # turn off with -d:TTT_CrucibleDisableConstFolding, e.g. for size sweeps
  when not defined(TTT_CrucibleDisableConstFolding):
    reg.register("foldConstantShells", pkTransform, phaseMain,
      "Dissolves calls to zero-arg constant-returning functions into the constant, deletes the dead definitions",
      dependsOn = @["mangleNames", "foldMaxMinToBuiltins"],
      run = foldConstantShells
    )


# Pass-by-ref materialization
# ═══════════════════════════════════════════════════════════════════

proc isLvalue*(n: GpuAst): bool =
  case n.kind
  of gpuIdent: true
  of gpuIndex: true
  of gpuDeref: true
  else: false

proc walkNonLvalueArgs(ctx: var GpuContext, n: var GpuAst) =
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
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    walkNonLvalueArgs(ctx, fn.pBody)


# Fold max/min idioms to backend-native builtins
# ═══════════════════════════════════════════════════════════════════

const
  BasicNumericKinds* = {gtUint8, gtUint16, gtInt16, gtUint32, gtInt32,
                        gtUint64, gtInt64, gtFloat32, gtFloat64, gtFloat16,
                        gtBf16, gtSize_t}

proc isBasicNumeric*(t: GpuType): bool =
  not t.isNil and t.kind in BasicNumericKinds

proc sameExpr(a, b: GpuAst): bool =
  ## Structural equality for pattern operands.
  ## `iSym` is the immutable fingerprint, the name may be mangled.
  ##
  ## Caution:
  ## - GpuAst has a custom `==`, structural, and it raiseAsserts on non-idents
  ## - never apply `==` or `!=` against nil here, use isNil
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

proc foldMinMaxPattern(ctx: GpuContext, n: var GpuAst): bool =
  ## Replaces a max/min-idiom gpuTernary with the backend-native builtin.
  ##
  ## - max folds `(B <= A) ? A : B`, min folds `(A <= B) ? A : B`
  ## - floats require the NaN-guard form `((P <= Q) || !(B == B)) ? ...`
  ## - IEEE fmax returns +0.0 for fmax(+0.0, -0.0), Nim's guard form
  ##   returns -0.0. Accepted, the hardware instruction is the point.
  if n.isNil: return false
  if n.kind != gpuTernary: return false
  if n.tCond.isNil: return false
  if n.tThen.isNil or n.tElse.isNil: return false
  let tThen = n.tThen
  let tElse = n.tElse

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

  let tA = ctx.getOperandType(tThen)
  let tB = ctx.getOperandType(tElse)
  if not isBasicNumeric(tA) or not isBasicNumeric(tB): return false
  if tA.kind != tB.kind: return false
  # the unguarded float form is not fmax, NaN propagates to tElse
  # ((y <= x) is false for NaN y)
  if not hasGuard and tA.kind in {gtFloat32, gtFloat64, gtFloat16, gtBf16}: return false

  var callName = GpuAst(kind: gpuIdent, symbol: newSymbol(builtin))
  n = GpuAst(kind: gpuCall, cName: callName,
             cArgs: @[tThen, tElse])
  return true

proc foldMaxMinInBody(ctx: var GpuContext, n: var GpuAst) =
  ## Hand-written recursion, walk's closure cannot capture the GpuContext value type
  discard ctx.foldMinMaxPattern(n)
  for ch in n.mitems:
    ctx.foldMaxMinInBody(ch)

proc foldMaxMinToBuiltins*(ctx: var GpuContext) =
  ## Lowers max/min ternaries to the backend-native max/min builtins,
  ## each printer emits its hardware instruction (CUDA: PTX max.*).
  ## Handles both direct and blitted `_blit_N = ternary` forms.
  ##
  ## TODO(abs): fold `(x < 0) ? -x : x`.
  ## - blocked on Nim's float abs body containing `when nimvm:`
  ## - needs a per-backend name table (CUDA fabsf/fabs, OpenCL fabs, GLSL/Vulkan abs, WGSL abs)
  ## - Int abs is already native (magic AbsI)
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    ctx.foldMaxMinInBody(fn.pBody)

# ═══════════════════════════════════════════════════════════════════
# Constant-shell elimination
# ═══════════════════════════════════════════════════════════════════

proc scanConstBindings(stmts: seq[GpuAst], bodyLocals: var HashSet[string], binding: var Table[string, GpuAst], returns: var seq[GpuAst]): bool =
  ## Collects the single-assignment local bindings of a candidate body.
  ## False when a statement disqualifies: calls, control flow,
  ## a local bound twice, an assignment to a non-ident or non-local target
  for s in stmts:
    case s.kind
    of gpuComment, gpuDiscard:
      discard
    of gpuVar:
      let name = s.vName.symbol.name
      if name in binding:
        return false
      bodyLocals.incl name
      if s.vInit.kind != gpuDiscard:
        binding[name] = s.vInit
    of gpuAssign:
      if s.aLeft.kind != gpuIdent:
        return false
      let name = s.aLeft.symbol.name
      if name notin bodyLocals:
        return false
      if name in binding:
        return false
      binding[name] = s.aRight
    of gpuBlock:
      if s.isExpr:
        return false
      if not scanConstBindings(s.statements, bodyLocals, binding, returns):
        return false
    of gpuReturn:
      returns.add s.rValue
    else:
      return false
  result = true

proc resolveConstant(n: GpuAst, binding: Table[string, GpuAst]): GpuAst =
  ## Returns the constant `n` denotes, or nil, purely syntactic: a literal,
  ## a wrapper over constants, a constant-constructor field, or a local
  ## resolved through the single-assignment binding
  if n.isNil:
    return nil
  case n.kind
  of gpuLit:
    result = n
  of gpuIdent:
    if n.symbol.name in binding:
      result = resolveConstant(binding[n.symbol.name], binding)
    else:
      result = nil
  of gpuPrefix:
    let v = resolveConstant(n.pVal, binding)
    if not v.isNil:
      result = GpuAst(kind: gpuPrefix, pOp: n.pOp, pVal: v)
  of gpuConv:
    let v = resolveConstant(n.convExpr, binding)
    if not v.isNil:
      result = GpuAst(kind: gpuConv, convTo: n.convTo, convExpr: v)
  of gpuCast:
    let v = resolveConstant(n.cExpr, binding)
    if not v.isNil:
      result = GpuAst(kind: gpuCast, cTo: n.cTo, cExpr: v)
  of gpuObjConstr:
    result = n
    for f in n.ocFields:
      if resolveConstant(f.value, binding).isNil:
        return nil
  of gpuArrayLit:
    result = n
    for v in n.aValues:
      if resolveConstant(v, binding).isNil:
        return nil
  of gpuDot:
    let parent = resolveConstant(n.dParent, binding)
    if not parent.isNil and parent.kind == gpuObjConstr:
      let fname = n.dField.symbol.name
      for f in parent.ocFields:
        if f.name == fname:
          return resolveConstant(f.value, binding)
    result = nil
  else:
    result = nil

proc foldedConstant(fn: GpuAst): GpuAst =
  ## Returns the constant a function computes when its body is
  ## a single constant expression, nil otherwise. Nim's semcheck drops the static
  ## params of fully-static instantiations and leaves a zero-arg shell.
  doAssert fn.kind == gpuProc, "Expected a gpuProc, got " & $fn.kind
  if attGlobal in fn.pAttributes:
    return nil
  if fn.pRetType.isNil or fn.pRetType.kind == gtVoid:
    return nil
  if fn.pBody.kind != gpuBlock:
    return nil
  var binding = initTable[string, GpuAst]()
  var returns: seq[GpuAst] = @[]
  var bodyLocals = initHashSet[string]()
  if not scanConstBindings(fn.pBody.statements, bodyLocals, binding, returns):
    return nil
  if returns.len != 1:
    return nil

  for v in binding.values:
    if resolveConstant(v, binding).isNil:
      return nil
  result = resolveConstant(returns[0], binding)

proc calleeParams(ctx: GpuContext, callee: GpuAst): seq[GpuParam] =
  ## Returns the params of `callee`.
  ## Empty when the callee is absent from the tables (builtins, operators),
  ## their params always pass by value.
  if callee in ctx.allFnTab:
    result = ctx.allFnTab[callee].pParams
  elif callee in ctx.genericInsts:
    result = ctx.genericInsts[callee].pParams
  else:
    result = @[]

proc substituteConstShells(ctx: GpuContext, n: var GpuAst, folds: Table[string, GpuAst]) =
  ## Replaces calls to folded constant functions with the constant.
  ## The pass keeps a call whole when an arg is not a literal or ident,
  ## the folded body reads no parameter. Reference-param args stay intact.
  if n.isNil:
    return
  case n.kind
  of gpuCall:
    let iSym = n.cName.symbol.iSym
    if iSym in folds:
      var argsSubstitutable = true
      for a in n.cArgs:
        if a.kind notin {gpuLit, gpuIdent}:
          argsSubstitutable = false
          break
      if argsSubstitutable:
        n = folds[iSym].clone()
        return
    let params = ctx.calleeParams(n.cName)
    for i in 0 ..< n.cArgs.len:
      if i < params.len and params[i].passByRef:
        continue
      ctx.substituteConstShells(n.cArgs[i], folds)
  else:
    for ch in n.mitems:
      ctx.substituteConstShells(ch, folds)


proc collectCallTargets(n: GpuAst, targets: var HashSet[string]) =
  if n.isNil:
    return
  if n.kind == gpuCall and not n.cName.isNil and not n.cName.symbol.isNil:
    targets.incl n.cName.symbol.iSym
  for ch in n.items:
    collectCallTargets(ch, targets)

proc foldConstantShells*(ctx: var GpuContext) =
  ## Dissolves calls to functions whose body is a single constant expression,
  ## deletes the now-dead definitions. Nim's semcheck turns a static-param
  ## function instantiated with static args into a zero-arg shell:
  ##
  ##   proc bar(): int = 16      # after semcheck of bar[16]()
  ##   result = 16
  ##   return result
  ##
  ## Without the pass, the emitter keeps that shell and every call to it
  ## alive in the generated shader. Detection is purely syntactic, a zero-arg
  ## single-constant body is always constant.
  var folds = initTable[string, GpuAst]()
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind != gpuProc:
      continue
    let constant = fn.foldedConstant()
    if not constant.isNil:
      folds[fn.pName.symbol.iSym] = constant
  if folds.len == 0:
    return
  for fnKey in toSeq(keys(ctx.allFnTab)):
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      ctx.substituteConstShells(fn.pBody, folds)
  for fnKey in toSeq(keys(ctx.genericInsts)):
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      ctx.substituteConstShells(fn.pBody, folds)
  for i, gb in ctx.globalBlocks.mpairs:
    ctx.substituteConstShells(ctx.globalBlocks[i], folds)
  var surviving = initHashSet[string]()
  for fnKey in keys(ctx.allFnTab):
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      collectCallTargets(fn.pBody, surviving)
  for fnKey in keys(ctx.genericInsts):
    let fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      collectCallTargets(fn.pBody, surviving)
  for gb in ctx.globalBlocks:
    collectCallTargets(gb, surviving)
  var dead: seq[GpuAst] = @[]
  for fnKey in keys(ctx.allFnTab):
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc and fn.pName.symbol.iSym in folds and
        fn.pName.symbol.iSym notin surviving:
      dead.add fnKey
  for k in dead:
    ctx.allFnTab.del k
    ctx.genericInsts.del k