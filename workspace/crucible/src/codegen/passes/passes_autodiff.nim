# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
# Autodiff backward derivation: kernel-level reverse mode over CIRv2
#
# ############################################################
#
# Backward derivation transposes a kernel's linearization into its vector-Jacobian product: the pass walks
# a kernel's statements in forward order and emits, per statement, gradient statements that propagate the output
# gradient to the kernel's inputs. Shadow adjoints start zero-filled, so the emitted statements accumulate additively.
#
# - Buffer gradients gather/scatter through a shadow adjoint buffer `d_<name>`,
#   one per buffer parameter the kernel touches.
# - Scalar gradients accumulate into per-index-and-thread partials
#   `d_<name>_partials`, and one generated single-thread reduce kernel sums
#   sums those partials into the scalar's 1-element gradient buffer.
# - The pass records the pair metadata in `m.adMetas` and the generated kernel
#   names in `m.adBwdKernels`, both projected from the derivation walk itself:
#   no second body walk, no suffix matching.
#
# Input contract: the loop idiom. After normalization the frontend desugars
# every supported `for` into the shape below, which the pass pattern-matches
# and re-emits the shape with the derived body in place of the forward body
# minus the trailing increment. Anything outside the idiom refuses:
#
#   │ <hoisted start/stop expressions>    # zero or more ordinary statements
#   │ (var i <intT> .)                    # loop variable, no initializer
#   │ (asgn i <start>)                    # start assignment before the while
#   │ (while (lt <intT> i <bound>)        # exclusive bound: int literal or kernel param
#   │   <body>                            # the loop variable never re-assigns
#   │   (asgn i (add <intT> i <step>)))   # last body statement, positive int-literal step
#
# Constructs outside the differentiable subset refuse with a message prefixed
# `bwd_diff: `, naming the construct and the kernel. The pass returns refusals
# in its string result, never as exceptions.

{.experimental: "codeReordering".}

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ../ir/cirv2_pretty
import ../autodiff_types
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerAutodiffPasses*(reg: var PassRegistry) =
  reg.register("autodiffBwdDerive", pkTransform,
    "Derives every kernel's backward kernel: shadow-adjoint gather/scatter " &
    "for buffer gradients, generated single-thread reduce kernels for scalar " &
    "gradients, refusals outside the differentiable subset",
    run = autodiffBwdDerive)

# ═══════════════════════════════════════════════════════════════════════
# Node-kind closure table
# ═══════════════════════════════════════════════════════════════════════
#
# Every CirExprKind and CirStmtKind is classified exactly once:
#
# - gradient: the derivation has a gradient rule for the kind
# - refuse: loud `bwd_diff:` refusal naming the kind
# - pass-through: no gradient semantics, zero gradient, or verbatim emission
#
# A value whose type is not a 32- or 64-bit float has an identically zero gradient,
# so its class is pass-through: int operands are primal and bool results
# contribute no gradient.
#
# kind           class         rule
# -------------  ------------  ------------------------------------------
# eNil           pass-through  constant, zero gradient
# eTrue/eFalse   pass-through  constant, zero gradient
# eSym           gradient      float scalar param reduces into partials, local to its definition, other symbols carry no gradient
# eIntLit        pass-through  constant, zero gradient
# eFloatLit      pass-through  constant, zero gradient
# eStrLit        pass-through  constant, zero gradient
# eDeref         refuse        value-position dereference has no gradient, index bases peel it instead
# eAt/ePat       gradient      buffer element read: scatter-add through the shadow adjoint
# eDot           gradient      coordinate-builtin read is thread identity (zero gradient), other member access has no rule
# eAdd/eSub      gradient      chain the gradient to both operands, `-` negates the right one
# eMul           gradient      each operand's gradient is the incoming gradient times the other operand
# eDiv           gradient      left gets g / right, right gets -(g * left) / (right * right)
# eMod           pass-through  int-typed mod, zero gradient
# eShl..eBitXor  pass-through  int-typed bitwise and shift ops, zero gradient
# eEq..eLe       pass-through  bool result, zero gradient
# eNot/eAnd/eOr  pass-through  bool result, zero gradient
# eNeg           gradient      negates the gradient
# eTernary       gradient      then-gradient gated by the condition, else-gradient by its negation
# eCast          refuse        a bitcast reinterprets bits, its gradient would be garbage
# eConv          gradient      float-to-float passes the gradient across widths, int/bool operands are primal
# eAddr          refuse        address-of moves the gradient off the differentiated value
# eSizeof        pass-through  compile-time constant, zero gradient
# eOconstr       refuse        aggregate constructions have no member gradients
# eAconstr       refuse        aggregate constructions have no member gradients
# eCall          gradient      whitelisted math builtins derive by mechanical rules, reductions re-reduce their seeds (reduceMax gates by the recomputed argmax), atomics and barriers derive in statement position only
# sVar/sConst    gradient      scalar locals are single-assignment and substituted by their definitions, non-scalars refuse
# sAsgn          gradient      a buffer-slot write seeds the gradient from `d_<buffer>`, a local write defines the substituted value
# sIte           gradient      branch-only gradients under the substituted condition
# sWhile         refuse        a while without the desugared-for preamble is data-dependent
# sRet           pass-through  the frontend's trailing fall-through `ret(nil)`, every other return refuses
# sDiscard       pass-through  no value
# sCall          gradient      barriers re-emit verbatim, atomics accumulate atomically into shadow adjoints, every other statement call is refused
# sAsm/sEmit     refuse        raw target-source injection has no gradient

const BwdDiffPrefix = "bwd_diff: "
  ## Every refusal message starts with this prefix. Tests and users grep
  ## the pass's returned diagnostics for it.

const NodeRefusalText: array[CirExprKind, string] = block:
  ## Fixed refusal text per refused expression kind.
  ## Empty for every kind with gradient or pass-through semantics. Both walks refuse
  ## through `refuseNodeKind`, so each fixed text has exactly one site.
  var t: array[CirExprKind, string]
  t[eDeref] = "value-position dereference is outside the differentiable subset"
  t[eCast] = "bitcasts have no gradient; use a float-to-float conversion"
  t[eAddr] = "address-of moves the gradient off the differentiated value"
  t[eOconstr] = "aggregate constructions have no member gradients"
  t[eAconstr] = "aggregate constructions have no member gradients"
  t

const StmtRefusalText: array[CirStmtKind, string] = block:
  ## Fixed refusal text per refused statement kind.
  ## Empty for every kind with gradient or pass-through semantics. Both walks refuse
  ## through `refuseStmtKind`, so each fixed text has exactly one site.
  var t: array[CirStmtKind, string]
  t[sWhile] =
    "a loop without the desugared-for preamble is data-dependent and has no static reverse traversal"
  t[sAsm] = "raw target-source injection is outside the differentiable subset"
  t[sEmit] = "raw target-source injection is outside the differentiable subset"
  t[sRet] = "mid-kernel returns have no reverse control flow"
  t

const MathWhitelist = ["exp", "ln", "sqrt", "rsqrt"]
  ## Math builtins with mechanical gradient rules, each unary over a float:
  ## exp: g*e, ln: g/arg, sqrt: g/(2*sqrt(arg)), rsqrt: -g/(2*arg*sqrt(arg)).
  ## Every other callee is refused: inline it before autodiffBwdDerive.

func isDiffFloat(t: CirType): bool =
  t != nil and t.kind == ctFloat and (t.bits == 32 or t.bits == 64)

func isCoordRead(e: CirExpr): bool =
  ## True for a coordinate-builtin read: `thread_position_in_threadgroup.x` or the flat
  ## `thread_index_in_threadgroup`. Thread identity is not a differentiable value:
  ## the derivation substitutes such reads untouched.
  e.kind == eDot and e.dotBase.kind == eSym and e.dotBase.builtin and
    e.dotBase.coordBuiltin != cbkNone

func isMathBuiltinCall(e: CirExpr): bool =
  e.callee.kind == eSym and e.callee.builtin and e.args.len == 1 and
    e.callee.name in MathWhitelist

func getBufferElemType(p: CirParam): CirType =
  ## Element type of a buffer parameter, nil when `p` is not one.
  ## Buffer parameters are `ptr UncheckedArray[T]` (an `aptr`) or `ptr array[N, T]`,
  ## a static-shape buffer under the pointer whose elements are read through
  ## `at(deref p, i)`.
  case p.typ.kind
  of ctAptr:
    result = p.typ.pElem
  of ctPtr:
    if p.typ.pElem != nil and p.typ.pElem.kind == ctArray:
      result = p.typ.pElem.aElem
  else:
    discard

func getBufferRoot(e: CirExpr): CirExpr =
  ## Buffer parameter symbol at the root of an element access, nil for any other
  ## root shape. Index and deref parents peel to their operands: a `ptr array` parameter
  ## reads elements through `at(deref p, i)`.
  result = e
  while result != nil and result.kind in {eAt, ePat}:
    result = if result.kind == eAt: result.atBase else: result.patBase
  while result != nil and result.kind == eDeref:
    result = result.derefOperand
  if result != nil and result.kind != eSym:
    result = nil

func isScopePrefix(defPath, usePath: seq[string]): bool =
  if defPath.len > usePath.len:
    return false
  for i in 0 ..< defPath.len:
    if defPath[i] != usePath[i]:
      return false
  result = true

func isLoopPreamble(ss: seq[CirStmt], i: int): bool =
  ## True when `ss[i ..< i+3]` is the frontend's desugared `for`: a nil-init loop-variable declaration,
  ## its start assignment, then the re-tested while.
  ss[i].kind == sVar and ss[i].init == nil and
    i + 2 < ss.len and
    ss[i + 1].kind == sAsgn and ss[i + 1].lhs.kind == eSym and
    ss[i + 1].lhs.name == ss[i].name and
    ss[i + 2].kind == sWhile

# ═══════════════════════════════════════════════════════════════════════
# Derivation state
# ═══════════════════════════════════════════════════════════════════════

type AdEnv = object
  ## Per-kernel derivation state: a refusal channel, parameter classification,
  ## a single-assignment local table with definition scopes, and gradient targets
  ## recorded while the backward body is emitted.
  fn: CirProc                       ## kernel in flight, refusals name it
  err: string                       ## the first refusal, empty when clean
  paramNames: HashSet[string]       ## kernel parameter names
  bufferParams: Table[string, CirParam]  ## buffer param name -> param
  ##
  localDefs: Table[string, CirExpr]          ## local name -> defining expression
  localDefPath: Table[string, seq[string]]   ## local name -> scope path at definition
  defCounts: Table[string, int]              ## local name -> defining events, more than one refuses
  loopVars: HashSet[string]                  ## loop variable names, exempt from read checks
  loopCount: int                             ## top-level loops seen so far, more than one refuses
  loopBound: CirExpr                         ## the index-loop bound, nil without a loop
  written: HashSet[string]                   ## buffer params the kernel writes
  readBufs: HashSet[string]                  ## buffer params the kernel reads
  slotSeen: HashSet[string]                  ## "buffer@index" keys, each slot written at most once
  ##
  scalarHits: HashSet[string]       ## float scalar params with gradient demand
  ambientVar: CirExpr               ## innermost loop variable read, nil outside loops
  dIdents: Table[string, CirExpr]   ## buffer name -> shadow-adjoint symbol
  threadCount: int                  ## thread product of the kernel's workgroup, unset axes count 1
  atomicMode: bool                  ## true while deriving an atomic statement's value
  atomicCallee: CirExpr             ## the forward atomic call's callee, reused for the backward's accumulation

proc refuse(env: var AdEnv, what, why: string) =
  ## Records the first refusal. The message names the construct and the kernel,
  ## greppable by the `bwd_diff: ` prefix.
  if env.err.len == 0:
    env.err = BwdDiffPrefix & refuseAt(env.fn, what & ": " & why)

proc refuseNodeKind(env: var AdEnv, e: CirExpr) =
  ## Refuses an expression kind through `NodeRefusalText`, one site per fixed
  ## text shared by the validation walk and the derivation walk.
  refuse(env, $e.kind, NodeRefusalText[e.kind])

proc refuseStmtKind(env: var AdEnv, s: CirStmt) =
  ## Refuses a statement kind through `StmtRefusalText`, one site per fixed
  ## text shared by the validation walk and the derivation walk.
  refuse(env, $s.kind, StmtRefusalText[s.kind])

# Validation is mutually recursive through the loop bodies (validateLoop re-enters validateStmts),
# a cycle that codeReordering's deferral does not lift, so the walk is forward-declared here.
proc validateStmts(env: var AdEnv, ss: seq[CirStmt], path: seq[string], loopDepth: int, nested: bool = false)

# ═══════════════════════════════════════════════════════════════════════
# Validation walk (one forward pass before any derivation)
# ═══════════════════════════════════════════════════════════════════════

proc recordBufferAccess(env: var AdEnv, access: CirExpr, isWrite: bool) =
  ## Records a buffer-slot access and validates the root of the access chain:
  ## the root must be a buffer parameter.
  let root = getBufferRoot(access)
  if root.isNil or root.name notin env.bufferParams:
    refuse(env, $access.kind, "index root is not a buffer parameter: " & renderExpr(access))
    return
  if isWrite:
    env.written.incl root.name
  else:
    env.readBufs.incl root.name

proc validateExpr(env: var AdEnv, e: CirExpr, path: seq[string]) =
  ## Validates an expression subtree and records local reads in def-use order.
  ## A local read with no definition so far refuses.
  if env.err.len > 0:
    return
  case e.kind
  of eNil, eTrue, eFalse, eIntLit, eFloatLit, eStrLit, eSizeof:
    discard
  of eSym:
    if not e.builtin and e.name notin env.paramNames and
        e.name notin env.loopVars and e.name notin env.localDefs:
      refuse(env, "eSym", "identifier '" & e.name & "' is read before it is defined")
  of eDeref, eCast, eAddr, eOconstr, eAconstr:
    refuseNodeKind(env, e)
  of eAt, ePat:
    let base = if e.kind == eAt: e.atBase else: e.patBase
    let idx = if e.kind == eAt: e.atIndex else: e.patIndex
    if base.kind == eDeref:
      validateExpr(env, base.derefOperand, path)
    else:
      validateExpr(env, base, path)
    validateExpr(env, idx, path)
    recordBufferAccess(env, e, isWrite = false)
  of eDot:
    if not isCoordRead(e):
      refuse(env, "eDot", "member access has no field-wise gradient: " & renderExpr(e))
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor,
     eEq, eNeq, eLt, eLe, eNot, eAnd, eOr:
    for ch in getExprChildren(e):
      validateExpr(env, ch, path)
  of eNeg:
    validateExpr(env, e.unOperand, path)
  of eTernary:
    validateExpr(env, e.cond, path)
    validateExpr(env, e.thenE, path)
    validateExpr(env, e.elseE, path)
  of eConv:
    validateExpr(env, e.castOperand, path)
  of eCall:
    let callee = e.callee
    if callee.kind == eSym and callee.builtin and
        callee.atomicsBuiltin != cbkNone:
      refuse(env, "eCall", "atomic builtin '" & callee.name & "' differentiates only as a statement, its returned old value has no rule")
    elif callee.kind == eSym and callee.builtin and
        callee.synchroBuiltin != cbkNone:
      discard
    elif callee.kind == eSym and callee.builtin and
        callee.reductionBuiltin != cbkNone:
      if e.args.len != 1:
        refuse(env, "eCall", "reduction builtin '" & callee.name & "' takes one argument")
      else:
        validateExpr(env, e.args[0], path)
    elif isMathBuiltinCall(e):
      validateExpr(env, e.args[0], path)
    else:
      let name = if callee.kind == eSym: callee.name else: renderExpr(callee)
      refuse(env, "eCall", "call to '" & name & "' is outside the differentiable subset; inline the callee before autodiffBwdDerive")

proc validateLocalDecl(env: var AdEnv, s: CirStmt, path: seq[string]) =
  ## Validates one local declaration: the differentiable subset takes
  ## explicitly initialized scalar locals, each defined exactly once.
  if not isScalarType(s.declType):
    refuse(env, $s.kind, "local '" & s.name & "' of type " & renderType(s.declType) & " is outside the differentiable subset; locals are scalars")
    return
  if s.init == nil:
    refuse(env, $s.kind, "local '" & s.name & "' declares no initializer; the differentiable subset takes explicitly initialized locals")
    return
  env.defCounts[s.name] = env.defCounts.getOrDefault(s.name, 0) + 1
  if env.defCounts[s.name] > 1:
    refuse(env, $s.kind, "local '" & s.name & "' is assigned more than once")
    return
  env.localDefs[s.name] = s.init
  env.localDefPath[s.name] = path
  validateExpr(env, s.init, path)

proc refuseAssignTarget(env: var AdEnv, s: CirStmt) =
  ## Refuses the assignment targets outside the differentiable subset:
  ## deref writes, field writes, and any other target shape.
  case s.lhs.kind
  of eDeref:
    refuse(env, "sAsgn", "deref writes are outside the differentiable subset; buffer writes go through an index")
  of eDot:
    refuse(env, "sAsgn", "field writes have no field-wise gradient")
  else:
    refuse(env, "sAsgn", "unsupported assignment target " & $s.lhs.kind)

proc validateAssign(env: var AdEnv, s: CirStmt, path: seq[string]) =
  ## Validates one assignment, a buffer-slot write or a local definition:
  ## each buffer slot is written at most once, each local is defined exactly once.
  case s.lhs.kind
  of ePat, eAt:
    recordBufferAccess(env, s.lhs, isWrite = true)
    if env.err.len > 0:
      return
    let idx = if s.lhs.kind == ePat: s.lhs.patIndex else: s.lhs.atIndex
    let slotKey = getBufferRoot(s.lhs).name & "@" & renderExpr(idx)
    if slotKey in env.slotSeen:
      refuse(env, "sAsgn", "buffer slot '" & slotKey & "' is written more than once")
      return
    env.slotSeen.incl slotKey
    validateExpr(env, idx, path)
    validateExpr(env, s.rhs, path)
  of eSym:
    let name = s.lhs.name
    if name in env.loopVars:
      refuse(env, "sAsgn", "the loop variable '" & name & "' is only mutated by the loop increment")
      return
    if name notin env.localDefs:
      refuse(env, "sAsgn", "assignment target '" & name & "' is not a local of this kernel")
      return
    env.defCounts[name] = env.defCounts.getOrDefault(name, 0) + 1
    if env.defCounts[name] > 1:
      refuse(env, "sAsgn", "assignment redefines local '" & name & "', loop-carried mutation is outside the subset")
      return
    env.localDefs[name] = s.rhs
    env.localDefPath[name] = path
    validateExpr(env, s.rhs, path)
  else:
    refuseAssignTarget(env, s)

proc validateAtomicStmt(env: var AdEnv, s: CirStmt, path: seq[string]) =
  ## Validates one atomic statement: the target is a buffer-slot address,
  ## recorded as a write. Atomic slots accumulate across threads, so the write-once
  ## slot rule does not apply.
  let calleeName = s.callCallee.name
  if s.callArgs.len != 2:
    refuse(env, "sCall", "atomic builtin '" & calleeName & "' takes a target address and a value")
    return
  let target = s.callArgs[0]
  if target.kind != eAddr or target.addrOperand.kind notin {ePat, eAt}:
    refuse(env, "sCall", "atomic builtin '" & calleeName & "' differentiates over a buffer-slot target, addr <buffer>[<index>]")
    return
  recordBufferAccess(env, target.addrOperand, isWrite = true)
  if env.err.len > 0:
    return
  let access = target.addrOperand
  let idx = if access.kind == ePat: access.patIndex else: access.atIndex
  validateExpr(env, idx, path)
  validateExpr(env, s.callArgs[1], path)

proc validateLoopPattern(env: var AdEnv, whl: CirStmt, varName: string) =
  ## Validates the loop pattern of one desugared-for while: the condition compares
  ## the loop variable on the left with a literal-or-param bound, a non-empty body,
  ## and the increment is the trailing positive-literal `i = i + <step>`.
  let cond = whl.whileCond
  if cond.kind != eLt:
    refuse(env, "sWhile", "the loop condition must be an exclusive < comparison over the loop variable")
    return
  if cond.a.kind != eSym or cond.a.name != varName:
    refuse(env, "sWhile", "the loop condition must compare the loop variable on the left")
    return
  let bound = cond.b
  if not (bound.kind == eIntLit or
          (bound.kind == eSym and bound.name in env.paramNames)):
    refuse(env, "sWhile", "the loop bound must be an int literal or a kernel parameter; the reduce trip count derives from it")
    return
  if whl.whileBody.len == 0:
    refuse(env, "sWhile", "the loop body is empty")
    return
  let incr = whl.whileBody[^1]
  if not (incr.kind == sAsgn and incr.lhs.kind == eSym and
          incr.lhs.name == varName and incr.rhs.kind == eAdd and
          incr.rhs.left.kind == eSym and incr.rhs.left.name == varName and
          incr.rhs.right.kind == eIntLit and incr.rhs.right.value > 0):
    refuse(env, "sWhile", "the loop increment must be the trailing i = i + <positive literal>")
    return

proc validateLoop(env: var AdEnv, ss: seq[CirStmt], i: int, path: seq[string], loopDepth: int) =
  ## Validates one desugared-for triple at `ss[i]`: a single top-level loop,
  ## then the shape checks of `validateLoopPattern`, then the start
  ## expression and the loop body.
  let varName = ss[i].name
  let whl = ss[i + 2]
  if loopDepth >= 1:
    refuse(env, "sWhile", "nested loops are outside the differentiable subset")
    return
  if env.loopCount == 1:
    refuse(env, "sWhile", "the kernel carries more than one top-level loop; the differentiable subset is a single index loop")
    return
  env.loopCount += 1
  env.loopVars.incl varName
  validateLoopPattern(env, whl, varName)
  if env.err.len > 0:
    return
  env.loopBound = whl.whileCond.b
  validateExpr(env, ss[i + 1].rhs, path)
  validateStmts(env, whl.whileBody[0 ..< whl.whileBody.len - 1],
                path & ("loop_" & varName), loopDepth + 1, nested = true)

proc validateStmts(env: var AdEnv, ss: seq[CirStmt], path: seq[string], loopDepth: int, nested: bool = false) =
  ## One forward walk over a statement list: the walk refuses the kinds
  ## outside the differentiable subset, records local definitions in def-use order,
  ## and classifies buffer accesses as written or read.
  ##
  ## `nested` marks a statement list below the kernel's top level: a return there
  ## leaves statement groups the forward never executed, the derivation refuses it.
  ## The top level's last statement may be the frontend's fall-through `ret(nil)`.
  if env.err.len > 0:
    return
  var i = 0
  while i < ss.len:
    if env.err.len > 0:
      return
    if isLoopPreamble(ss, i):
      validateLoop(env, ss, i, path, loopDepth)
      i += 3
      continue
    let s = ss[i]
    case s.kind
    of sVar, sConst:
      validateLocalDecl(env, s, path)
    of sAsgn:
      validateAssign(env, s, path)
    of sIte:
      validateExpr(env, s.cond, path)
      validateStmts(env, s.thenBody, path & "then", loopDepth, nested = true)
      validateStmts(env, s.elseBody, path & "else", loopDepth, nested = true)
    of sWhile:
      refuseStmtKind(env, s)
    of sRet:
      if s.retVal != nil or nested or i != ss.len - 1:
        refuseStmtKind(env, s)
    of sDiscard:
      discard
    of sCall:
      let callee = s.callCallee
      if callee.kind == eSym and callee.builtin and
          callee.synchroBuiltin != cbkNone:
        # barrier: the backward keeps the same barrier at the same position
        discard
      elif callee.kind == eSym and callee.builtin and
          callee.atomicsBuiltin != cbkNone:
        validateAtomicStmt(env, s, path)
      else:
        let name = if callee.kind == eSym: callee.name else: renderExpr(callee)
        refuse(env, "sCall", "call to '" & name & "' as a statement is outside the differentiable subset")
    of sAsm, sEmit:
      refuseStmtKind(env, s)
    i += 1

# ═══════════════════════════════════════════════════════════════════════
# Local substitution
# ═══════════════════════════════════════════════════════════════════════

proc substWalk(env: var AdEnv, e: CirExpr, path: seq[string]) =
  ## In-place local substitution over a fresh clone: a local read becomes
  ## its defining expression, and the expression's own locals substitute in turn.
  ## A local whose definition scope does not enclose the use scope refuses.
  if e.kind == eSym:
    if e.name in env.localDefs:
      if not isScopePrefix(env.localDefPath.getOrDefault(e.name, @[]), path):
        refuse(env, "eSym", "local '" & e.name & "' escapes its defining scope into a gradient")
        return
      let def = cloneExpr(env.localDefs[e.name])
      substWalk(env, def, path)
      e[] = def[]
    return
  for ch in getExprChildren(e):
    substWalk(env, ch, path)

proc substituteLocals(env: var AdEnv, e: CirExpr, path: seq[string]): CirExpr =
  ## Returns a fresh copy of `e` with every in-scope local read replaced by
  ## its defining expression, so the backward's conditions and index
  ## expressions are self-contained: the backward declares no forward locals.
  ## The forward tree is never touched.
  if e.isNil:
    return nil
  result = cloneExpr(e)
  substWalk(env, result, path)

# ═══════════════════════════════════════════════════════════════════════
# Gradient construction helpers
# ═══════════════════════════════════════════════════════════════════════

func shadowAccess(access: CirExpr): CirExpr =
  ## Shadow-adjoint form of a buffer element access: a clone whose root symbol
  ## is the buffer's `d_<name>` shadow, same access shape.
  result = cloneExpr(access)
  let root = getBufferRoot(result)
  spliceSym(root, "d_" & root.name, root.typ)

func mathCall(name: string, arg: CirExpr, ret: CirType): CirExpr =
  ## Re-emits one whitelisted math builtin inside a gradient expression over
  ## `arg`, and the printers dispatch the canonical name.
  callE(sym(name, tProcT([getExprType(arg)], ret), builtin = true), arg)

func reduceCallee(name: string, argTyp: CirType, kind: CirReductionBuiltinKind): CirExpr =
  ## Re-emits a subgroup reduction in the backward over the same lane set,
  ## with the forward's thread mapping.
  result = sym(name, tProcT([argTyp], argTyp), builtin = true)
  result.reductionBuiltin = kind

func threadIndexRead(intTyp: CirType): CirExpr =
  ## Flat thread-index builtin read for the per-thread term of a multi-thread
  ## workgroup's partials slot.
  result = sym("thread_index_in_threadgroup", intTyp, builtin = true)
  result.coordBuiltin = cbkThreadIndexInThreadgroup

# ═══════════════════════════════════════════════════════════════════════
# Expression derivation
# ═══════════════════════════════════════════════════════════════════════

proc deriveExpr(env: var AdEnv, e: CirExpr, g: CirExpr, elemTyp: CirType, path: seq[string], into: var seq[CirStmt]) =
  ## Emits the gradient statements that propagate the adjoint `g` of `e`
  ## into `e`'s differentiable inputs, in forward order. A value whose type
  ## is not a 32- or 64-bit float has an identically zero gradient, and the walk
  ## returns before emitting anything for it.
  if env.err.len > 0:
    return
  let t = getExprType(e)
  if t.isNil or not isDiffFloat(t):
    return
  case e.kind
  of eAdd:
    deriveExpr(env, e.left, g, elemTyp, path, into)
    deriveExpr(env, e.right, g, elemTyp, path, into)
  of eSub:
    deriveExpr(env, e.left, g, elemTyp, path, into)
    deriveExpr(env, e.right, negE(e.opType, g), elemTyp, path, into)
  of eMul:
    deriveExpr(env, e.left, arithE(eMul, e.opType, g, e.right), elemTyp, path, into)
    deriveExpr(env, e.right, arithE(eMul, e.opType, g, e.left), elemTyp, path, into)
  of eDiv:
    deriveExpr(env, e.left, arithE(eDiv, e.opType, g, e.right), elemTyp, path, into)
    let denom = arithE(eMul, e.opType, e.right, e.right)
    let dYdX = arithE(eDiv, e.opType, arithE(eMul, e.opType, g, e.left), denom)
    deriveExpr(env, e.right, negE(e.opType, dYdX), elemTyp, path, into)
  of ePat, eAt:
    # Buffer read: scatter-add the adjoint into the buffer's shadow adjoint.
    # Under an atomic statement the forward scattered into the slot atomically
    # and threads share the slot, so the backward accumulates atomically too.
    # A direct add would race across threads.
    let accessSub = substituteLocals(env, e, path)
    if env.err.len > 0:
      return
    let gSub = substituteLocals(env, g, path)
    if env.err.len > 0:
      return
    let lhs = shadowAccess(accessSub)
    if env.atomicMode:
      let addrTyp = tPtr(resolveAddressSpace(lhs), getExprType(lhs))
      into.add callS(cloneExpr(env.atomicCallee),
                     @[addrE(addrTyp, lhs), gSub])
    else:
      let rhs = shadowAccess(accessSub)
      into.add asgnS(lhs, arithE(eAdd, getExprType(lhs), rhs, gSub))
  of eSym:
    if e.name in env.paramNames:
      # Scalar gradient: one partial per primal loop index, reduced later.
      if env.ambientVar.isNil:
        refuse(env, "eSym", "scalar gradient of '" & e.name & "' arises outside the index loop")
        return
      if e.name notin env.scalarHits:
        env.scalarHits.incl e.name
      let intTyp = getExprType(env.ambientVar)
      let slotIdx = if env.threadCount > 1:
        arithE(eAdd, intTyp,
               arithE(eMul, intTyp, env.ambientVar,
                      intLit(env.threadCount, intTyp)),
               threadIndexRead(intTyp))
      else:
        env.ambientVar
      let partialsTyp = tAptr(caDevice, t)
      let lhs = patE(sym("d_" & e.name & "_partials", partialsTyp), slotIdx)
      let rhs = patE(sym("d_" & e.name & "_partials", partialsTyp), slotIdx)
      let gSub = substituteLocals(env, g, path)
      if env.err.len > 0:
        return
      let gSeed = if sameType(elemTyp, t): gSub else: convE(t, gSub)
      into.add asgnS(lhs, arithE(eAdd, t, rhs, gSeed))
    elif e.name in env.localDefs:
      if not isScopePrefix(env.localDefPath.getOrDefault(e.name, @[]), path):
        refuse(env, "eSym", "gradient position reads local '" & e.name & "' outside its defining scope")
        return
      deriveExpr(env, env.localDefs[e.name], g, elemTyp, path, into)
    else:
      refuse(env, "eSym", "float value '" & e.name & "' has no definition to derive")
  of eNeg:
    deriveExpr(env, e.unOperand, negE(e.unType, g), elemTyp, path, into)
  of eTernary:
    let cond = substituteLocals(env, e.cond, path)
    if env.err.len > 0:
      return
    deriveExpr(env, e.thenE,
               ternaryE(t, cond, g, fltLit(0.0, t)), elemTyp, path, into)
    deriveExpr(env, e.elseE,
               ternaryE(t, cond, fltLit(0.0, t), g), elemTyp, path, into)
  of eConv:
    let innerT = getExprType(e.castOperand)
    if innerT.isNil or not isDiffFloat(innerT):
      # int and bool operands are primal (loop index, int param), zero gradient
      return
    let gInner = if sameType(innerT, t): g else: convE(innerT, g)
    deriveExpr(env, e.castOperand, gInner, elemTyp, path, into)
  of eCall:
    let callee = e.callee
    if callee.kind == eSym and callee.builtin and
        callee.reductionBuiltin != cbkNone:
      # Reduction derivation: the reduction value's incoming gradient is the cross-lane
      # sum of the per-lane seeds. reduceAdd passes that sum to every contributing lane.
      # reduceMax derives through the argmax mask: the backward recomputes the primal max
      # and the argmax lane alone receives the sum, every other lane receives zero.
      let arg = e.args[0]
      let gSub = substituteLocals(env, g, path)
      if env.err.len > 0:
        return
      let gSum = callE(reduceCallee("reduceAdd", t, cbkReduceAdd), gSub)
      if callee.reductionBuiltin == cbkReduceAdd:
        deriveExpr(env, arg, gSum, elemTyp, path, into)
      else:
        let argSub = substituteLocals(env, arg, path)
        if env.err.len > 0:
          return
        let primalMax = callE(reduceCallee(callee.name, t, cbkReduceMax), argSub)
        let gate = ternaryE(t, eqE(argSub, primalMax),
                            fltLit(1.0, t), fltLit(0.0, t))
        deriveExpr(env, arg, arithE(eMul, t, gSum, gate), elemTyp, path, into)
    elif isMathBuiltinCall(e):
      let arg = e.args[0]
      let argSub = substituteLocals(env, arg, path)
      if env.err.len > 0:
        return
      case callee.name
      of "exp":
        deriveExpr(env, arg, arithE(eMul, t, g, mathCall("exp", argSub, t)),
                   elemTyp, path, into)
      of "ln":
        deriveExpr(env, arg, arithE(eDiv, t, g, argSub), elemTyp, path, into)
      of "sqrt":
        let denom = arithE(eMul, t, fltLit(2.0, t), mathCall("sqrt", argSub, t))
        deriveExpr(env, arg, arithE(eDiv, t, g, denom), elemTyp, path, into)
      of "rsqrt":
        let twoX = arithE(eMul, t, fltLit(2.0, t), argSub)
        let denom = arithE(eMul, t, twoX, mathCall("sqrt", argSub, t))
        deriveExpr(env, arg, negE(t, arithE(eDiv, t, g, denom)),
                   elemTyp, path, into)
      else:
        refuse(env, "eCall", "no gradient rule for the math builtin '" & callee.name & "', inline the callee before autodiffBwdDerive")
    else:
      let name = if callee.kind == eSym: callee.name else: renderExpr(callee)
      refuse(env, "eCall", "the gradient of a call to '" & name & "' has no rule; inline the callee before autodiffBwdDerive")
  of eDot:
    if not isCoordRead(e):
      refuse(env, "eDot", "the gradient of member access " & renderExpr(e) & " has no rule to transpose")
    # a coordinate read is thread identity, zero gradient
  of eCast, eAddr, eDeref,
     eOconstr, eAconstr:
    refuseNodeKind(env, e)
  of eNil, eTrue, eFalse, eIntLit, eFloatLit, eStrLit, eSizeof,
     eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor,
     eEq, eNeq, eLt, eLe, eNot, eAnd, eOr:
    discard

# ═══════════════════════════════════════════════════════════════════════
# Statement derivation
# ═══════════════════════════════════════════════════════════════════════

proc deriveStmts(env: var AdEnv, ss: seq[CirStmt], path: seq[string]): seq[CirStmt] =
  ## Gradient groups of a statement list in forward order. The groups accumulate
  ## additively into zero-filled shadow buffers, so the group order does not
  ## change the sums.
  if env.err.len > 0:
    return
  var i = 0
  while i < ss.len:
    if env.err.len > 0:
      return
    if isLoopPreamble(ss, i):
      # re-emit the desugared-for preamble and while with the derived body
      # in place of the forward body, minus the trailing increment
      let varName = ss[i].name
      let whl = ss[i + 2]
      let saved = env.ambientVar
      env.ambientVar = sym(varName, ss[i].declType)
      let inner = deriveStmts(env, whl.whileBody[0 ..< whl.whileBody.len - 1],
                              path & ("loop_" & varName))
      env.ambientVar = saved
      if env.err.len > 0:
        return
      if inner.len > 0:
        result.add cloneStmts(@[ss[i], ss[i + 1]])
        result.add whileS(cloneExpr(whl.whileCond),
                          inner & cloneStmts(@[whl.whileBody[^1]]),
                          whl.whileUnroll)
      i += 3
      continue
    let s = ss[i]
    case s.kind
    of sAsgn:
      if s.lhs.kind in {ePat, eAt}:
        let seedTyp = getExprType(s.lhs)
        if isDiffFloat(seedTyp):
          deriveExpr(env, s.rhs, shadowAccess(s.lhs), seedTyp, path, result)
      # a local definition derives at its read sites, nothing emits here
    of sIte:
      let thenS = deriveStmts(env, s.thenBody, path & "then")
      let elseS = deriveStmts(env, s.elseBody, path & "else")
      if env.err.len > 0:
        return
      if thenS.len > 0 or elseS.len > 0:
        let cond = substituteLocals(env, s.cond, path)
        if env.err.len > 0:
          return
        result.add iteS(cond, thenS, elseS)
    of sCall:
      let callee = s.callCallee
      if callee.kind == eSym and callee.builtin and
          callee.synchroBuiltin != cbkNone:
        # barrier: structural identity, so the backward emits the same sync structure
        # at the same statement position
        result.add cloneStmts(@[s])[0]
      elif callee.kind == eSym and callee.builtin and
          callee.atomicsBuiltin != cbkNone:
        # atomic backward: the value's gradient is the target slot's incoming gradient,
        # and the scatters that the value induces accumulate atomically into the shadow
        # adjoints of the touched buffers
        let access = s.callArgs[0].addrOperand
        let seedTyp = getExprType(access)
        if isDiffFloat(seedTyp):
          let g = shadowAccess(access)
          env.atomicCallee = s.callCallee
          env.atomicMode = true
          deriveExpr(env, s.callArgs[1], g, seedTyp, path, result)
          env.atomicMode = false
      else:
        let name = if callee.kind == eSym: callee.name else: renderExpr(callee)
        refuse(env, "sCall", "gradient position has no rule for the statement call to '" & name & "'")
    of sVar, sConst, sDiscard:
      discard
    of sRet:
      if s.retVal != nil:
        refuseStmtKind(env, s)
    of sWhile:
      refuseStmtKind(env, s)
    of sAsm, sEmit:
      refuseStmtKind(env, s)
    i += 1

# ═══════════════════════════════════════════════════════════════════════
# Kernel assembly
# ═══════════════════════════════════════════════════════════════════════

proc makeReduceKernel(env: AdEnv, fn: CirProc, p: CirParam): CirProc =
  ## Generates the single-thread reduce kernel of one scalar gradient: the kernel sums
  ## the per-index-and-thread partials from the backward kernel's workspace
  ## in ascending slot order and writes the sum into the scalar's 1-element
  ## gradient buffer `d_<name>`.
  let elemTyp = p.typ
  let intTyp = getExprType(env.loopBound)
  let partialsTyp = tAptr(caDevice, elemTyp)
  let outTyp = tAptr(caDevice, elemTyp)
  let iName = p.name & "_reduce_i"
  let accName = p.name & "_reduce_acc"
  let iE = sym(iName, intTyp)
  let accE = sym(accName, elemTyp)
  let partialsE = sym("d_" & p.name & "_partials", partialsTyp)
  let outE = sym("d_" & p.name, outTyp)
  let nE = sym("n", intTyp)
  let zeroIdx = intLit(0, intTyp)
  let body = @[
    varD(accName, elemTyp, fltLit(0.0, elemTyp)),
    varD(iName, intTyp),
    asgnS(iE, intLit(0, intTyp)),
    whileS(ltE(iE, nE), @[
      asgnS(accE, arithE(eAdd, elemTyp, accE, patE(partialsE, iE))),
      asgnS(iE, arithE(eAdd, intTyp, iE, intLit(1, intTyp))),
    ]),
    asgnS(patE(outE, zeroIdx), arithE(eAdd, elemTyp, patE(outE, zeroIdx), accE)),
    retS(nil),
  ]
  result = CirProc(
    name: fn.name & "_bwd_d_" & p.name & "_reduce",
    params: @[param("d_" & p.name & "_partials", partialsTyp),
              param("n", intTyp),
              param("d_" & p.name, outTyp)],
    ret: tVoid(),
    body: body,
    isKernel: true)

proc recordAdMetas(m: CirModule, fn: CirProc, env: AdEnv) =
  ## Single metadata derivation of one kernel pair, projected from the same
  ## AdEnv that built the backward kernel's parameter list. The pair metadata
  ## pipeline consumes this record: no second body walk, no suffix matching,
  ## so the metadata and the launched kernels cannot desync.
  var metas: seq[BwdParamMeta]
  for p in fn.params:
    if getBufferElemType(p) != nil:
      var meta = BwdParamMeta(name: p.name)
      meta.kind = if p.name in env.written: pkResidual else: pkBuffer
      if p.name in env.dIdents:
        meta.gradKind = gkBuffer
        meta.gradName = "d_" & p.name
      if p.typ.kind == ctPtr and p.typ.pElem.kind == ctArray:
        meta.extent = xeStaticShape
        meta.staticLen = p.typ.pElem.aN
      elif env.loopBound != nil and env.loopBound.kind == eSym and
          env.loopBound.name in env.paramNames:
        meta.extent = xeLenParam
        meta.extentParam = env.loopBound.name
      else:
        meta.extent = xeWrapperField
      metas.add meta
    elif isDiffFloat(p.typ):
      var meta = BwdParamMeta(name: p.name, kind: pkScalar)
      if p.name in env.scalarHits:
        meta.gradKind = gkBuffer1
        meta.gradName = "d_" & p.name
      metas.add meta
    else:
      metas.add BwdParamMeta(name: p.name, kind: pkScalar)

  var workspaces: seq[ScalarWorkspace]
  for meta in metas:
    if meta.gradKind == gkBuffer1:
      var tripCount = -1
      var tripCountParam = ""
      var tripCount64 = false
      if env.loopBound.kind == eIntLit:
        tripCount = env.loopBound.value.int
        tripCount64 = env.loopBound.litType.bits >= 64
      else:
        tripCountParam = env.loopBound.name
        tripCount64 = getExprType(env.loopBound).bits >= 64
      workspaces.add ScalarWorkspace(
        gradName: meta.gradName,
        workspaceName: "d_" & meta.name & "_partials",
        tripCount: tripCount,
        tripCountParam: tripCountParam,
        tripCount64: tripCount64,
        threadCount: env.threadCount)
  m.adMetas[fn.name] = AdKernelsMetas(params: metas, workspaces: workspaces)

proc deriveKernel(m: CirModule, fn: CirProc): string =
  ## Derives one kernel's backward kernel plus one single-thread reduce kernel
  ## per scalar gradient. The forward kernel stays as it is.
  var env = AdEnv(fn: fn)
  for p in fn.params:
    env.paramNames.incl p.name
    if getBufferElemType(p) != nil:
      env.bufferParams[p.name] = p
  env.threadCount = fn.workgroup[0].max(1) * fn.workgroup[1].max(1) *
                    fn.workgroup[2].max(1)

  validateStmts(env, fn.body, @[], 0)
  if env.err.len > 0:
    return env.err
  if env.written.len == 0:
    refuse(env, "kernel", "writes no buffer, nothing to differentiate")
    return env.err
  for p in fn.params:
    if p.name in env.written and p.name in env.readBufs:
      refuse(env, "kernel", "reads buffer '" & p.name & "' it also writes; the differentiable subset is out-of-place")
      return env.err

  # Shadow-adjoint symbols: d_<buffer> for every buffer the kernel touches.
  for p in fn.params:
    if p.name in env.written or p.name in env.readBufs:
      env.dIdents[p.name] = sym("d_" & p.name, p.typ)

  let groups = deriveStmts(env, fn.body, @[])
  if env.err.len > 0:
    return env.err

  # Backward signature: the d-block first, then the gradients of the forward
  # params in forward order, then the needed forward params, then the reduce workspaces.
  var needed: HashSet[string]
  let onExpr = proc(e: CirExpr) =
    if e.kind == eSym:
      needed.incl e.name
  walkStmts(groups, nil, onExpr)

  var params: seq[CirParam]
  for p in fn.params:
    if p.name in env.dIdents:
      params.add param("d_" & p.name, p.typ)
  for p in fn.params:
    if p.name in needed:
      params.add p
  var reduceKernels: seq[CirProc]
  for p in fn.params:
    if isDiffFloat(p.typ) and p.name in env.scalarHits:
      params.add param("d_" & p.name & "_partials", tAptr(caDevice, p.typ))
      reduceKernels.add makeReduceKernel(env, fn, p)

  let bwd = CirProc(
    name: fn.name & "_bwd",
    params: params,
    ret: tVoid(),
    body: groups & @[retS(nil)],
    isKernel: true,
    workgroup: fn.workgroup)
  m.procs.add bwd
  for rk in reduceKernels:
    m.procs.add rk
  m.adBwdKernels.incl bwd.name
  for rk in reduceKernels:
    m.adBwdKernels.incl rk.name
  recordAdMetas(m, fn, env)
  return ""

proc autodiffBwdDerive(m: CirModule): string =
  ## Pass entry: derives the backward of every kernel in the module. Kernels that the derivation
  ## itself generated (backward and reduce kernels) are recorded in
  ## `m.adBwdKernels` and never re-derived, so a user-declared backward
  ## pre-registered there is left untouched.
  var kernels: seq[CirProc]
  for fn in m.procs:
    if fn.isKernel and fn.name notin m.adBwdKernels:
      kernels.add fn
  for fn in kernels:
    let err = deriveKernel(m, fn)
    if err.len > 0:
      return err
  return ""
