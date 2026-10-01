## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
import std / [sequtils, sets, tables]
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
    ## Types the backends' native max/min accept.
    ## Int[N] structs (gtObject) are deliberately excluded.
    ## ceramic's genBinOp handles those.

proc isBasicNumeric*(t: GpuType): bool =
  ## Returns whether `t` is a type the backends' native max/min accept.
  not t.isNil and t.kind in BasicNumericKinds

proc operandType*(ctx: GpuContext; n: GpuAst): GpuType =
  ## Best-effort type of a pattern operand.
  ## Returns nil when the pass cannot determine it,
  ## and the fold then conservatively skips.
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.operandType(n.pVal)  # `-x` keeps x's type
  of gpuCall: ctx.getFnReturnType(n.cName)
  else: nil

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

proc foldMinMaxPattern(ctx: GpuContext; n: var GpuAst): bool =
  ## If `n` is a gpuTernary matching the max/min idiom, replaces the node
  ## with a call to the backend-native max/min builtin.
  ##
  ## The builtin is spelled with the plain ambiguous-builtin name that every
  ## backend supports for basic types.
  ##
  ## Matched shapes, operands A = tThen and B = tElse.
  ##
  ## Max folds `(B <= A) ? A : B`, or on floats
  ## `((B <= A) || !(B == B)) ? A : B` with the NaN guard.
  ##
  ## Min folds `(A <= B) ? A : B`, or on floats
  ## `((A <= B) || !(B == B)) ? A : B` with the NaN guard.
  ##
  ## Known semantic trade-off:
  ## - the ternary is Nim's exact semantics
  ## - IEEE fmax returns +0.0 for fmax(+0.0, -0.0)
  ## - Nim's guard form returns -0.0 for that input, accepted
  ##
  ## The hardware instruction is the point of the pass.
  if n.isNil: return false
  if n.kind != gpuTernary: return false
  if n.tCond.isNil: return false
  if n.tThen.isNil or n.tElse.isNil: return false
  let tThen = n.tThen
  let tElse = n.tElse

  # The condition is Le(P, Q), optionally OR'd with a NaN guard !(X == X).
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

  # Both operands must be basic numerics of the same type.
  let tA = ctx.operandType(tThen)
  let tB = ctx.operandType(tElse)
  if not isBasicNumeric(tA) or not isBasicNumeric(tB): return false
  if tA.kind != tB.kind: return false
  # The unguarded form on floats is not fmax.
  # NaN propagates to tElse, as (y <= x) is false when y is NaN,
  # so y is returned. Only the NaN-guarded form is fmax-equivalent on floats.
  if not hasGuard and tA.kind in {gtFloat32, gtFloat64, gtFloat16, gtBf16}: return false

  var callName = GpuAst(kind: gpuIdent, symbol: newSymbol(builtin))
  n = GpuAst(kind: gpuCall, cIsExpr: true, cName: callName,
             cArgs: @[tThen, tElse])
  return true

proc foldMaxMinInBody(ctx: var GpuContext; n: var GpuAst) =
  ## Recursive pre-order walker.
  ## `walk`'s closure cannot capture the GpuContext value type,
  ## hence the hand-written recursion.
  ##
  ## foldMinMaxPattern may replace `n` with a gpuCall whose args are the same operand refs.
  ## Recursing into the children still visits those operand expressions.
  discard ctx.foldMinMaxPattern(n)
  for ch in n.mitems:
    ctx.foldMaxMinInBody(ch)

proc foldMaxMinToBuiltins*(ctx: var GpuContext) =
  ## Rewrites ternary max/min idioms to the backend-native max/min builtins,
  ## so they lower to the PTX `max.f32` / `max.u32` / `max.s32` hardware instructions.
  ##
  ## Returns true when a rewrite happened, false otherwise.
  ##
  ## The fold is a pure expression rewrite.
  ## It replaces the ternary subtree in place, handling both forms:
  ## - direct, `result = ternary`
  ## - blitted, `_blit_N = ternary`, afterwards `result = _blit_N`
  ##
  ## The rewritten device functions stay contained.
  ## The backends inline them.
  ##
  ## TODO(abs):
  ## - fold `(x < 0) ? -x : x` to the backend abs builtin too
  ##
  ## Float abs is blocked:
  ## - Nim's float abs body contains `when nimvm:`,
  ##   which crashes translation before any pass sees it
  ## - a per-backend name table is needed
  ##   (CUDA fabsf/fabs, OpenCL fabs, GLSL/Vulkan abs, WGSL abs)
  ##
  ## Int abs is uniform (abs), already native (magic AbsI),
  ## so the table is floats-only.
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    ctx.foldMaxMinInBody(fn.pBody)

# ═══════════════════════════════════════════════════════════════════
# Constant-shell elimination
# ═══════════════════════════════════════════════════════════════════

proc scanConstBindings(stmts: seq[GpuAst], binding: var Table[string, GpuAst], returns: var seq[GpuAst]): bool =
  ## Collects the single-assignment local bindings of a candidate body.
  ##
  ## Returns:
  ## - true when every statement can belong to a constant body
  ## - false otherwise
  ##
  ## Disqualifying statements:
  ## - any statement kind other than comments, discard, local declarations,
  ##   plain assignments, plain scope blocks and returns
  ##
  ## The disqualifying kinds also rule out side effects,
  ## since calls and control statements are among them.
  ## - any local bound twice
  ## - any assignment whose target is not a plain identifier
  for s in stmts:
    case s.kind
    of gpuComment, gpuDiscard:
      discard
    of gpuVar:
      let name = s.vName.symbol.name
      if name in binding:
        return false
      if s.vInit.kind != gpuDiscard:
        binding[name] = s.vInit
    of gpuAssign:
      if s.aLeft.kind != gpuIdent:
        return false
      let name = s.aLeft.symbol.name
      if name in binding:
        return false
      binding[name] = s.aRight
    of gpuBlock:
      if s.isExpr:
        return false
      if not scanConstBindings(s.statements, binding, returns):
        return false
    of gpuReturn:
      returns.add s.rValue
    else:
      return false
  result = true

proc resolveConstant(n: GpuAst; binding: Table[string, GpuAst]): GpuAst =
  ## Returns the constant expression `n` denotes, or nil when `n` is not constant, determined purely syntactically.
  ##
  ## Constant forms:
  ## - a literal
  ## - a prefix, conversion, cast or constructor literal over constants
  ## - field access into a constant constructor,
  ##   or a local resolved through the single-assignment `binding`
  ##
  ## An identifier outside `binding` (a parameter, a global) is not constant.
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
    # Reading a field of a constant constructor is that field's value.
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
  ## Returns the constant a function computes when its body is a single
  ## constant expression, or nil otherwise.
  ##
  ## Candidate shape:
  ## - no parameters at all, or parameters the body never reads
  ## - a non-void return type
  ##
  ## Nim's semcheck drops the static parameters of a fully-static instantiation, leaving a zero-arg shell.
  ## In both candidate shapes the returned value cannot depend on the arguments.
  ##
  ## The body must reduce to exactly one constant along
  ## the single-assignment chain feeding `return`.
  doAssert fn.kind == gpuProc, "Expected a gpuProc, got " & $fn.kind
  if attGlobal in fn.pAttributes:
    return nil
  if fn.pRetType.isNil or fn.pRetType.kind == gtVoid:
    return nil
  if fn.pBody.kind != gpuBlock:
    return nil
  var binding = initTable[string, GpuAst]()
  var returns: seq[GpuAst] = @[]
  if not scanConstBindings(fn.pBody.statements, binding, returns):
    return nil
  if returns.len != 1:
    return nil
  result = resolveConstant(returns[0], binding)

proc calleeParams(ctx: GpuContext; callee: GpuAst): seq[GpuParam] =
  ## Parameters of `callee`.
  ##
  ## Returns an empty seq when the callee is unknown to the tables.
  ## Unknown callees are builtins and operators, whose parameters
  ## are always by value.
  if callee in ctx.allFnTab:
    result = ctx.allFnTab[callee].pParams
  elif callee in ctx.genericInsts:
    result = ctx.genericInsts[callee].pParams
  else:
    result = @[]

proc substituteConstShells(ctx: GpuContext, n: var GpuAst, folds: Table[string, GpuAst]) =
  ## Replaces every call to a folded constant function with the constant.
  ##
  ## Arguments bound to reference parameters are left intact.
  ## A substituted constant is not an lvalue, and reference parameters
  ## reject temporaries, so only the non-lvalue risk is avoided.
  if n.isNil:
    return
  case n.kind
  of gpuCall:
    let iSym = n.cName.symbol.iSym
    if iSym in folds:
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


proc collectCallTargets(n: GpuAst; targets: var HashSet[string]) =
  ## Collects the iSym of every function called in a subtree.
  if n.isNil:
    return
  if n.kind == gpuCall and not n.cName.isNil and not n.cName.symbol.isNil:
    targets.incl n.cName.symbol.iSym
  for ch in n.items:
    collectCallTargets(ch, targets)

proc foldConstantShells*(ctx: var GpuContext) =
  ## Dissolves calls to functions whose body is a single constant expression,
  ## and deletes the now-dead definitions.
  ##
  ## Precondition:
  ## `ctx` holds a fully translated IR, post semcheck.
  ##
  ## Effects:
  ## - each call to such a function is replaced by the constant it computes
  ## - definitions left without callers are deleted
  ##
  ## Nim's semcheck turns a static-param function instantiated with static
  ## arguments into a zero-arg function whose body is one constant,
  ## `result = 16`, then `return result`.
  ##
  ## Without the pass, the emitter keeps that shell and every call to it alive in the generated shader.
  ##
  ## Detection is purely syntactic.
  ## A zero-argument single-constant body IS constant, so a false positive is impossible.
  ## See `foldedConstant` for the exact shape.
  ##
  ## A definition survives only if a call survives. That happens exactly
  ## when a call site binds a non-scalar constant to a reference parameter.
  ## See `substituteConstShells` for that case.
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
  # A definition is dead iff no call to it remains.
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

proc registerOptimizationPasses*(reg: var PassRegistry) =
  ## Register optimization passes. Runs after preprocessing (mangleNames)
  ## so constructed builtin calls keep their plain name.
  reg.register("foldMaxMinToBuiltins", pkTransform, phaseMain,
    "Folds ternary max/min idioms to backend-native max/min builtins",
    dependsOn = @["mangleNames"],
    run = foldMaxMinToBuiltins
  )
  # Turn off the pass with the define TTT_CrucibleDisableConstFolding,
  # for example to measure the pass against the pre-pass codegen in size sweeps.
  when not defined(TTT_CrucibleDisableConstFolding):
    reg.register("foldConstantShells", pkTransform, phaseMain,
      "Dissolves calls to zero-arg constant-returning functions into the constant, deletes the dead definitions",
      dependsOn = @["mangleNames", "foldMaxMinToBuiltins"],
      run = foldConstantShells
    )