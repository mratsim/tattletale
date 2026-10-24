# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
# Autodiff transposer: kernel-level reverse mode over GpuAst
#
# ############################################################
#
# Transposes one post-pass GpuAst kernel into its backward kernel.
# Buffer cotangents are gather/scatter statements through the shadow
# adjoint of the primal buffer.
# Scalar cotangents are per-index partials that a generated axis-reduce
# kernel sums into the scalar's gradient buffer.
# Constructs outside the differentiable subset refuse at compile time
# with a `bwd_diff:` prefixed error.

{.experimental: "codeReordering".}

import std / [macros, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_helpers
import ./pass_registry

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerAutodiffPasses*(reg: var PassRegistry) =
  reg.register("autodiffTranspose", pkTransform, phaseMain,
    "Transposes every global kernel into its backward kernel: shadow-adjoint " &
    "gather/scatter for buffer cotangents, generated axis-reduce kernels for " &
    "scalar cotangents, loud refusals outside the differentiable subset",
    dependsOn = @[],
    run = autodiffTranspose
  )

# ═══════════════════════════════════════════════════════════════════════
# Node-kind closure table
# ═══════════════════════════════════════════════════════════════════════
#
# Every GpuNodeKind is classified exactly once:
#
# - local-rule: the transposer has a cotangent rule for the kind
# - refuse: loud `bwd_diff:` compile-time error
# - pass-through: no gradient semantics
#
# The table is the audit contract of the differentiable subset.
# The refusal suite locks one red test per row (tests/codegen/autodiff/).
#
# kind           class         rule
# -------------  ------------  ------------------------------------------
# gpuDiscard     pass-through  empty statement, no gradient semantics
# gpuProc        refuse        nested proc definition in a kernel body
# gpuCall        local-rule    external math builtins {exp, ln/log, sqrt, rsqrt} transpose with mechanical cotangent rules, kernel-defined callees and every other call refuse (inline the callee before bwd_diff), barrier/atomic/reduction builtins refuse with their own message
# gpuIf          local-rule    branch-only cotangents, each branch body transposes under the matching condition, the condition itself stays primal
# gpuTernary     local-rule    then-cotangent gated by the condition, else-cotangent gated by its negation
# gpuFor         local-rule    loop structure reused verbatim, bodies are iteration-independent by contract (loop-carried overwrites refuse), cotangent sums are order-independent
# gpuWhile       refuse        data-dependent trip count has no static reverse traversal
# gpuBinOp       local-rule    + - * / transposition over primitive floats, comparisons refuse as differentiated values
# gpuVar         local-rule    single-assignment locals are substituted by their definition in cotangent expressions, pointer/array/view locals refuse
# gpuAssign      local-rule    a buffer-slot write seeds the cotangent from d_<y>, a local write defines the substituted value, compound operators refuse (in-place update)
# gpuIdent       local-rule    scalar param to generated reduce partials, local to substitution, int/bool to zero cotangent, whole buffer to refuse
# gpuLit         pass-through  constants have zero cotangent
# gpuArrayLit    refuse        aggregate literals have no member cotangents in v1
# gpuPrefix      local-rule    `-` negates the cotangent, boolean prefixes carry zero cotangent
# gpuBlock       local-rule    transparent container in statement position, expression-position blocks refuse
# gpuReturn      refuse        mid-kernel return has no reverse control flow
# gpuDot         refuse        member access has no field-wise cotangent in v1, coordinate-builtin reads imply thread parallelism and refuse (the subset is single-thread)
# gpuIndex       local-rule    gather/scatter through the shadow adjoint of the indexed buffer
# gpuTypeDef     pass-through  type definitions live in ctx.types
# gpuAlias       pass-through  alias declarations carry no values
# gpuObjConstr   refuse        object construction has no field-wise cotangents in v1
# gpuInlineAsm   refuse        opaque assembly has no cotangent
# gpuEmit        refuse        raw target-source injection has no cotangent
# gpuAddr        refuse        address-of moves the gradient off the differentiated value
# gpuDeref       local-rule    pointer-peel form of an index parent only, value-position derefs refuse
# gpuConv        local-rule    float-to-float conversion passes the cotangent through (narrowed back across widths), conversions to int/bool of differentiated values refuse
# gpuCast        refuse        a bitcast reinterprets bits, its cotangent would be garbage, gpuConv is the type-change rule
# gpuComment     pass-through  comments carry no values
# gpuConstexpr   pass-through  compile-time constants, lifted before AD
# gpuMaterialize local-rule    materialization wrapper, the cotangent passes through to the wrapped expression

const BwdDiffPrefix = "bwd_diff: "
  ## Every refusal message starts with this prefix. Tests and users grep
  ## compile-time diagnostics for it.

const DiffFloatKinds = {gtFloat32, gtFloat64}
  ## Float widths that carry cotangents in v1.

func isTransposableBinOp(op: string): bool =
  ## Primitive float arithmetic operators, each with a transposition rule.
  case op
  of "+", "-", "*", "/": true
  else: false

func isComparisonBinOp(op: string): bool =
  ## Comparison operators, refused as differentiated values.
  case op
  of "<", ">", "<=", ">=", "==", "!=": true
  else: false

const MathBuiltinWhitelist = ["exp", "ln", "log", "sqrt", "rsqrt"]
  ## Mechanical cotangent rules for the whitelisted external math builtins:
  ## `exp`, `ln` (also `log`), `sqrt`, `rsqrt`, each unary over a float.
  ## Defined callees and the base-taking `log` overload refuse: inline the callee before bwd_diff.

func isMathBuiltinCall(n: GpuAst): bool =
  ## True for a unary call to one of the whitelisted math builtins.
  if n.cName.kind != gpuIdent or n.cArgs.len != 1:
    return false
  result = n.cName.ident() in MathBuiltinWhitelist

# ═══════════════════════════════════════════════════════════════════════
# Refusals
# ═══════════════════════════════════════════════════════════════════════

proc adRefuse(n: GpuAst, what, why: string) =
  ## Loud compile-time refusal, the only diagnostic channel of the pass.
  ## The message starts with the prefix and names the offending construct.
  error BwdDiffPrefix & what & ": " & why

proc refuseNodeKind(n: GpuAst) =
  ## Refusals for the kinds with no cotangent rule, one branch per refused
  ## row of the node-kind closure table above.
  case n.kind
  of gpuProc:
    adRefuse(n, "gpuProc",
      "nested proc definitions are outside the differentiable subset")
  of gpuWhile:
    adRefuse(n, "gpuWhile",
      "data-dependent loops have no static reverse traversal")
  of gpuReturn:
    adRefuse(n, "gpuReturn",
      "mid-kernel returns have no reverse control flow")
  of gpuArrayLit:
    adRefuse(n, "gpuArrayLit",
      "aggregate literals have no member cotangents in v1")
  of gpuObjConstr:
    adRefuse(n, "gpuObjConstr",
      "object construction has no field-wise cotangents in v1")
  of gpuInlineAsm:
    adRefuse(n, "gpuInlineAsm",
      "inline assembly is outside the differentiable subset")
  of gpuEmit:
    adRefuse(n, "gpuEmit",
      "raw target-source injection is outside the differentiable subset")
  of gpuAddr:
    adRefuse(n, "gpuAddr",
      "address-of moves the gradient off the differentiated value")
  of gpuCast:
    adRefuse(n, "gpuCast",
      "bitcasts have no cotangent; use a float-to-float gpuConv")
  else:
    adRefuse(n, $n.kind, "unexpected node in the transposer")

# ═══════════════════════════════════════════════════════════════════════
# Transposer state
# ═══════════════════════════════════════════════════════════════════════

type AdEnv = object
  ## Per-kernel transposer state:
  ##
  ## - parameter classification (buffer vs float-scalar params)
  ## - the single-assignment local table with definition scopes
  ## - cotangent targets built while the backward body is emitted
  bufferParams: Table[string, GpuParam]     ## buffer param iSym -> param
  scalarParams: Table[string, GpuParam]     ## float scalar param iSym -> param
  localDefs: Table[string, GpuAst]          ## local iSym -> its defining expression
  localDefPath: Table[string, seq[string]]  ## local iSym -> scope path at definition
  defCounts: Table[string, int]             ## local iSym -> number of defining events
  loopVars: HashSet[string]                 ## loop variable iSyms, exempt from def-use checks
  ##
  written: HashSet[string]                  ## buffer param iSyms the kernel writes
  readBufs: HashSet[string]                 ## buffer param iSyms the kernel reads
  scalarHits: seq[string]                   ## scalar param iSyms with cotangent demand
  scalarLoop: string                        ## loop scope hosting scalar contributions, "" = none
  scalarLoopVar: GpuAst                     ## that loop's variable
  scalarLoopCount: GpuAst                   ## that loop's trip-count expression
  ambientVar: GpuAst                        ## innermost loop variable, nil outside loops
  ambientCount: GpuAst                      ## innermost loop's trip-count expression
  currentLoopScope: string                  ## scope id of the innermost loop, "" outside
  ##
  dIdents: Table[string, GpuAst]            ## buffer param iSym -> its shadow-adjoint ident
  partials: Table[string, GpuAst]           ## scalar param iSym -> its reduce-partials ident
  slotSeen: HashSet[string]                 ## "buffer@index" keys, each slot written at most once

func scopePush(path: seq[string], id: string): seq[string] =
  ## Extends a scope path with a nested statement-list identity.
  result = path
  result.add id

func isScopePrefix(defPath, usePath: seq[string]): bool =
  ## True when a definition scope encloses a use scope.
  if defPath.len > usePath.len:
    return false
  for i in 0 ..< defPath.len:
    if defPath[i] != usePath[i]:
      return false
  result = true

# ═══════════════════════════════════════════════════════════════════════
# Validation + local-definition scan (one forward walk)
# ═══════════════════════════════════════════════════════════════════════

proc indexRootParam(n: GpuAst): GpuAst =
  ## Parameter ident at the root of an index chain, nil when the root is
  ## not a plain ident. A gpuDeref parent peels to its operand.
  if n.kind == gpuIdent:
    return n
  if n.kind == gpuDeref:
    return indexRootParam(n.dOf)
  result = nil

proc recordBufferUse(n: GpuAst, env: var AdEnv, isWrite: bool) =
  ## Records a buffer-slot access and validates its root.
  let root = indexRootParam(n.iArr)
  if root.isNil or root.symbol.isNil:
    adRefuse(n, "gpuIndex", "index root is not a buffer parameter")
  let sym = root.symbol
  if sym.typ.isNil or sym.typ.kind != gtPtr:
    adRefuse(n, "gpuIndex",
      "index root '" & sym.name & "' is not a buffer parameter")
  if sym.iSym in env.defCounts:
    adRefuse(n, "gpuIndex",
      "index root '" & sym.name & "' is a local; local arrays and pointer " &
      "locals are outside the differentiable subset")
  if isWrite:
    env.written.incl sym.iSym
  else:
    env.readBufs.incl sym.iSym

proc validateExpr(ctx: var GpuContext, n: GpuAst, env: var AdEnv, path: seq[string]) =
  ## Validates an expression subtree and records local reads in def-use order.
  ## A local read with no definition so far refuses.
  case n.kind
  of gpuWhile, gpuReturn, gpuArrayLit, gpuObjConstr, gpuInlineAsm,
     gpuEmit, gpuAddr, gpuCast, gpuProc:
    refuseNodeKind(n)
  of gpuCall:
    let nameIdent = n.cName
    if nameIdent.kind == gpuIdent and nameIdent.symbol != nil:
      if nameIdent.symbol.synchroBuiltin != gbkNone:
        adRefuse(n, "gpuCall",
          "synchronization builtin '" & nameIdent.symbol.name &
          "' is outside the differentiable subset")
      if nameIdent.symbol.atomicsBuiltin != gbkNone:
        adRefuse(n, "gpuCall",
          "atomic builtin '" & nameIdent.symbol.name &
          "' is outside the differentiable subset (no atomics)")
    if not isMathBuiltinCall(n):
      adRefuse(n, "gpuCall",
        "call to '" & nameIdent.ident() &
        "' is outside the differentiable subset; inline the callee before bwd_diff")
    validateExpr(ctx, n.cArgs[0], env, path)
  of gpuDot:
    if n.dParent.kind == gpuIdent and n.dParent.symbol != nil:
      if n.dParent.symbol.coordBuiltin != gbkNone:
        adRefuse(n, "gpuDot",
          "coordinate builtin read '" & n.dParent.ident() & "." &
          n.dField.ident() & "' implies thread parallelism; the differentiable " &
          "subset launches one thread, move the traversal into a loop")
      else:
        adRefuse(n, "gpuDot",
          "member access '" & n.dParent.ident() & "." & n.dField.ident() &
          "' has no field-wise cotangent in v1")
  of gpuDeref:
    adRefuse(n, "gpuDeref",
      "value-position dereference is outside the differentiable subset")
  of gpuVar:
    adRefuse(n, "gpuVar", "declaration in expression position")
  of gpuIdent:
    let sym = n.symbol
    if sym != nil and sym.symKind == gsLocal and
        sym.iSym notin env.loopVars and
        env.defCounts.getOrDefault(sym.iSym, 0) == 0:
      adRefuse(n, "gpuIdent",
        "local '" & sym.name & "' is read before it is assigned")
  of gpuIndex:
    if n.iArr.kind == gpuDeref:
      validateExpr(ctx, n.iArr.dOf, env, path)
    else:
      validateExpr(ctx, n.iArr, env, path)
    validateExpr(ctx, n.iIndex, env, path)
    recordBufferUse(n, env, isWrite = false)
  else:
    for ch in n.items:
      validateExpr(ctx, ch, env, path)

proc validateAssign(ctx: var GpuContext, n: GpuAst, env: var AdEnv, path: seq[string]) =
  ## Validates one assignment:
  ##
  ## - target shape (buffer slot or local)
  ## - compound operators refuse (in-place update)
  ## - each buffer slot is written at most once
  if n.aOp != "=":
    adRefuse(n, "gpuAssign",
      "compound assignment '" & n.aOp & "' is an in-place update")
  case n.aLeft.kind
  of gpuIndex:
    recordBufferUse(n.aLeft, env, isWrite = true)
    let slotKey = indexRootParam(n.aLeft.iArr).ident() & "@" &
                  n.aLeft.iIndex.pretty()
    if slotKey in env.slotSeen:
      adRefuse(n, "gpuAssign",
        "buffer slot '" & slotKey & "' is written more than once")
    env.slotSeen.incl slotKey
    validateExpr(ctx, n.aLeft.iIndex, env, path)
    validateExpr(ctx, n.aRight, env, path)
  of gpuIdent:
    let sym = n.aLeft.symbol
    if sym.iSym notin env.defCounts and sym.iSym notin env.loopVars:
      adRefuse(n, "gpuAssign",
        "assignment target '" & sym.name & "' is not a local of this kernel")
    env.defCounts[sym.iSym] = env.defCounts.getOrDefault(sym.iSym, 0) + 1
    if env.defCounts[sym.iSym] > 1:
      adRefuse(n, "gpuAssign",
        "local '" & sym.name & "' is assigned more than once " &
        "(loop-carried mutation)")
    env.localDefs[sym.iSym] = n.aRight
    env.localDefPath[sym.iSym] = path
    validateExpr(ctx, n.aRight, env, path)
  else:
    adRefuse(n, "gpuAssign", "unsupported assignment target " & $n.aLeft.kind)

proc validateStmts(ctx: var GpuContext, stmts: seq[GpuAst], env: var AdEnv, path: seq[string], loopDepth: int) =
  ## One forward walk over a statement list:
  ##
  ## - refuses the kinds outside the differentiable subset
  ## - records local definitions in def-use order
  ## - classifies buffer accesses (written vs read)
  for s in stmts:
    case s.kind
    of gpuAssign:
      validateAssign(ctx, s, env, path)
    of gpuVar:
      let t = s.vType
      if t.kind in {gtPtr, gtUncheckedArray, gtArray, gtSpan, gtObject,
                    gtGenericInst, gtVoidPtr, gtString}:
        adRefuse(s, "gpuVar",
          "local '" & s.vName.ident() & "' of type " & pretty(t) &
          " is outside the differentiable subset; locals are scalars in v1")
      if s.vInit != nil and s.vInit.kind != gpuDiscard:
        let sym = s.vName.symbol
        env.defCounts[sym.iSym] = env.defCounts.getOrDefault(sym.iSym, 0) + 1
        if env.defCounts[sym.iSym] > 1:
          adRefuse(s, "gpuVar",
            "local '" & sym.name & "' is assigned more than once")
        env.localDefs[sym.iSym] = s.vInit
        env.localDefPath[sym.iSym] = path
        validateExpr(ctx, s.vInit, env, path)
    of gpuFor:
      if loopDepth >= 1:
        adRefuse(s, "gpuFor", "nested loops are outside the differentiable subset")
      if s.fRangeKind != rkExclusive:
        adRefuse(s, "gpuFor",
          "only exclusive ranges (a ..< b) carry a static trip count")
      env.loopVars.incl s.fVar.symbol.iSym
      validateExpr(ctx, s.fStart, env, path)
      validateExpr(ctx, s.fEnd, env, path)
      validateExpr(ctx, s.fStep, env, path)
      validateStmts(ctx, s.fBody.statements, env,
                    path.scopePush("for" & s.fVar.symbol.iSym), loopDepth + 1)
    of gpuIf:
      validateExpr(ctx, s.ifCond, env, path)
      validateStmts(ctx, s.ifThen.statements, env,
                    path.scopePush("then"), loopDepth)
      for i in 0 ..< s.ifElifs.len:
        let el = s.ifElifs[i]
        validateExpr(ctx, el.cond, env, path)
        validateStmts(ctx, el.body.statements, env,
                      path.scopePush("elif" & $i), loopDepth)
      if s.ifElse.kind != gpuDiscard:
        validateStmts(ctx, s.ifElse.statements, env,
                      path.scopePush("else"), loopDepth)
    of gpuBlock:
      validateStmts(ctx, s.statements, env, path, loopDepth)
    of gpuDiscard, gpuComment, gpuConstexpr:
      discard
    of gpuMaterialize:
      validateExpr(ctx, s.mExpr, env, path)
    else:
      refuseNodeKind(s)

# ═══════════════════════════════════════════════════════════════════════
# Cotangent construction helpers
# ═══════════════════════════════════════════════════════════════════════

func mkBinOp(op: string, l, r: GpuAst, t: GpuType): GpuAst =
  ## Builds a primitive binary operation over floats.
  result = GpuAst(kind: gpuBinOp,
                  bOp: newGpuIdent(op),
                  bLeft: l, bRight: r,
                  bIsOverloaded: false,
                  bType: t)

func mkPrefix(op: string, v: GpuAst): GpuAst =
  GpuAst(kind: gpuPrefix, pOp: op, pVal: v)

func mkIndex(arr, idx: GpuAst): GpuAst =
  GpuAst(kind: gpuIndex, iArr: arr, iIndex: idx)

func mkAssign(l, r: GpuAst): GpuAst =
  GpuAst(kind: gpuAssign, aLeft: l, aRight: r, aOp: "=")

func mkZeroLit(t: GpuType): GpuAst =
  GpuAst(kind: gpuLit, lValue: "0", lType: t)

func mkConv(toType: GpuType, e: GpuAst): GpuAst =
  GpuAst(kind: gpuConv, convTo: toType, convExpr: e)

func mkBufferPtr(elemTyp: GpuType): GpuType =
  ## Type of a kernel buffer parameter over `elemTyp`:
  ## `ptr UncheckedArray[elemTyp]` in the device address space.
  GpuType(kind: gtPtr,
          to: GpuType(kind: gtUncheckedArray, uaTo: elemTyp),
          addressSpace: asDevice)

proc mathCall(ctx: var GpuContext, name: string, arg: GpuAst, elemTyp: GpuType): GpuAst =
  ## Re-emits one whitelisted math builtin inside a cotangent expression,
  ## over `arg`. Registers the callee so type queries resolve.
  let cName = GpuAst(kind: gpuIdent,
                     symbol: newSymbol(name, iSym = name, symKind = gsProc))
  if cName notin ctx.builtinFns:
    ctx.builtinFns[cName] = GpuAst(kind: gpuProc, pName: cName,
                                   pRetType: elemTyp, pAttributes: {attDevice})
  result = GpuAst(kind: gpuCall, cName: cName, cArgs: @[arg])

func mkPartialsIdent(sym: Symbol, elemTyp: GpuType): GpuAst =
  ## Reduce-workspace ident of a scalar cotangent. It carries one element
  ## per primal loop index and is named after the scalar it reduces.
  let typ = mkBufferPtr(elemTyp)
  GpuAst(kind: gpuIdent,
         symbol: newSymbol("d_" & sym.name & "_partials",
                           iSym = "d_" & sym.iSym & "_partials",
                           typ = typ, symKind = gsDeviceKernelParam))

proc substituteLocals(ctx: var GpuContext, n: GpuAst, env: AdEnv, path: seq[string]): GpuAst =
  ## Rewrites local reads into their defining expressions so the backward
  ## kernel's conditions and index expressions are self-contained.
  ## Pure expression rewriting, no cotangent semantics.
  ## A local whose definition scope does not enclose the use scope refuses.
  case n.kind
  of gpuIdent:
    let sym = n.symbol
    if sym != nil and sym.iSym in env.localDefs:
      let defPath = env.localDefPath.getOrDefault(sym.iSym, @[])
      if not isScopePrefix(defPath, path):
        adRefuse(n, "gpuIdent",
          "local '" & sym.name & "' escapes its defining scope into a cotangent")
      return substituteLocals(ctx, env.localDefs[sym.iSym], env, path)
    return n
  of gpuBinOp:
    result = GpuAst(kind: gpuBinOp,
                    bOp: n.bOp,
                    bLeft: substituteLocals(ctx, n.bLeft, env, path),
                    bRight: substituteLocals(ctx, n.bRight, env, path),
                    bIsOverloaded: n.bIsOverloaded,
                    bType: n.bType)
  of gpuPrefix:
    result = GpuAst(kind: gpuPrefix, pOp: n.pOp,
                    pVal: substituteLocals(ctx, n.pVal, env, path))
  of gpuIndex:
    result = mkIndex(substituteLocals(ctx, n.iArr, env, path),
                     substituteLocals(ctx, n.iIndex, env, path))
  of gpuTernary:
    result = GpuAst(kind: gpuTernary,
                    tCond: substituteLocals(ctx, n.tCond, env, path),
                    tThen: substituteLocals(ctx, n.tThen, env, path),
                    tElse: substituteLocals(ctx, n.tElse, env, path))
  of gpuConv:
    result = mkConv(n.convTo, substituteLocals(ctx, n.convExpr, env, path))
  of gpuCall:
    result = GpuAst(kind: gpuCall, cName: n.cName)
    for a in n.cArgs:
      result.cArgs.add substituteLocals(ctx, a, env, path)
  of gpuLit, gpuDeref:
    result = n
  else:
    result = n

# ═══════════════════════════════════════════════════════════════════════
# Expression transposition
# ═══════════════════════════════════════════════════════════════════════

proc transposeExpr(ctx: var GpuContext, e: GpuAst, g: GpuAst, elemTyp: GpuType, env: var AdEnv, path: seq[string], into: var seq[GpuAst]) =
  ## Emits the cotangent statements that propagate the adjoint `g` of `e`
  ## into `e`'s differentiable inputs. Shadow buffers start zero-filled,
  ## so the statements accumulate additively:
  ##
  ## | Expression               | Cotangent                                                |
  ## | ------------------------ | -------------------------------------------------------- |
  ## | buffer read `x[i]`       | scatter-add, `d_x[i] = d_x[i] + g`                       |
  ## | scalar param             | per-index partial for the generated reduce kernel        |
  ## | whitelisted math builtin | transpose the argument by the builtin's mechanical rule, |
  ## |                          | recomputing the primal call in the cotangent expression  |
  ## | single-assignment local  | substitute the definition, also inside cotangent         |
  ## |                          | expressions; the backward declares no forward locals     |
  ## | int/bool subexpression   | identically zero cotangent                               |
  case e.kind
  of gpuBinOp:
    let op = e.bOp.ident()
    if isComparisonBinOp(op):
      adRefuse(e, "gpuBinOp",
        "comparison '" & op & "' used as a differentiated value has no gradient")
    let t = ctx.getExprType(e)
    if t.isNil or t.kind notin DiffFloatKinds:
      return
    if e.bIsOverloaded:
      adRefuse(e, "gpuBinOp",
        "overloaded operator '" & op & "' over non-primitive types is " &
        "outside the differentiable subset")
    if not isTransposableBinOp(op):
      adRefuse(e, "gpuBinOp", "operator '" & op & "' has no transposition rule")
    case op
    of "+":
      transposeExpr(ctx, e.bLeft, g, elemTyp, env, path, into)
      transposeExpr(ctx, e.bRight, g, elemTyp, env, path, into)
    of "-":
      transposeExpr(ctx, e.bLeft, g, elemTyp, env, path, into)
      transposeExpr(ctx, e.bRight, mkPrefix("-", g), elemTyp, env, path, into)
    of "*":
      transposeExpr(ctx, e.bLeft, mkBinOp("*", g, e.bRight, elemTyp),
                    elemTyp, env, path, into)
      transposeExpr(ctx, e.bRight, mkBinOp("*", g, e.bLeft, elemTyp),
                    elemTyp, env, path, into)
    of "/":
      transposeExpr(ctx, e.bLeft, mkBinOp("/", g, e.bRight, elemTyp),
                    elemTyp, env, path, into)
      let dYdX = mkBinOp("/", mkBinOp("*", g, e.bLeft, elemTyp),
                         mkBinOp("*", e.bRight, e.bRight, elemTyp), elemTyp)
      transposeExpr(ctx, e.bRight, mkPrefix("-", dYdX), elemTyp, env, path, into)
    else:
      discard
  of gpuPrefix:
    # getExprType has no gpuPrefix case: a prefix shares its operand's type
    let t = ctx.getExprType(e.pVal)
    if t.isNil or t.kind notin DiffFloatKinds:
      return
    if e.pOp == "-":
      transposeExpr(ctx, e.pVal, mkPrefix("-", g), elemTyp, env, path, into)
    else:
      adRefuse(e, "gpuPrefix",
        "prefix '" & e.pOp & "' over a float value has no transposition rule")
  of gpuLit:
    discard
  of gpuIdent:
    let sym = e.symbol
    if sym.isNil or sym.typ.isNil:
      adRefuse(e, "gpuIdent", "untyped identifier in a differentiated expression")
    case sym.typ.kind
    of DiffFloatKinds:
      if sym.iSym in env.scalarParams:
        # Scalar cotangent: one partial per primal loop index, reduced later.
        if env.ambientVar.isNil:
          adRefuse(e, "gpuIdent",
            "scalar cotangent of '" & sym.name &
            "' arises outside the index loop")
        if env.scalarLoop.len > 0 and env.currentLoopScope != env.scalarLoop:
          adRefuse(e, "gpuIdent",
            "scalar cotangent of '" & sym.name &
            "' arises in more than one loop")
        if env.scalarLoop.len == 0:
          env.scalarLoop = env.currentLoopScope
          env.scalarLoopVar = env.ambientVar
          env.scalarLoopCount = env.ambientCount
        if sym.iSym notin env.scalarHits:
          env.scalarHits.add sym.iSym
        let partials = env.partials.mgetOrPut(sym.iSym,
                                              mkPartialsIdent(sym, elemTyp))
        let slot = mkIndex(partials, env.ambientVar)
        let gSub = substituteLocals(ctx, g, env, path)
        into.add mkAssign(slot,
          mkBinOp("+", mkIndex(partials, env.ambientVar), gSub, elemTyp))
      elif sym.iSym in env.localDefs:
        if not isScopePrefix(env.localDefPath.getOrDefault(sym.iSym, @[]), path):
          adRefuse(e, "gpuIdent",
            "local '" & sym.name & "' escapes its defining scope into a cotangent")
        transposeExpr(ctx, env.localDefs[sym.iSym], g, elemTyp, env, path, into)
      else:
        adRefuse(e, "gpuIdent",
          "float value '" & sym.name & "' has no definition to transpose")
    of gtBool, gtInt16, gtInt32, gtInt64, gtUint16, gtUint32, gtUint64, gtSize_t:
      discard
    of gtPtr:
      adRefuse(e, "gpuIdent",
        "whole-buffer value '" & sym.name &
        "' in a differentiated expression has no element cotangent")
    else:
      adRefuse(e, "gpuIdent",
        "value of type " & pretty(sym.typ) & " is outside the differentiable subset")
  of gpuCall:
    let t = ctx.getExprType(e)
    if t.isNil or t.kind notin DiffFloatKinds:
      return
    let arg = e.cArgs[0]
    case e.cName.ident()
    of "exp":
      transposeExpr(ctx, arg, mkBinOp("*", g, e, elemTyp), elemTyp, env, path, into)
    of "log", "ln":
      transposeExpr(ctx, arg, mkBinOp("/", g, arg, elemTyp), elemTyp, env, path, into)
    of "sqrt":
      let two = GpuAst(kind: gpuLit, lValue: "2.0", lType: elemTyp)
      let denom = mkBinOp("*", two, ctx.mathCall("sqrt", arg, elemTyp), elemTyp)
      transposeExpr(ctx, arg, mkBinOp("/", g, denom, elemTyp), elemTyp, env, path, into)
    of "rsqrt":
      let two = GpuAst(kind: gpuLit, lValue: "2.0", lType: elemTyp)
      let twoX = mkBinOp("*", two, arg, elemTyp)
      let twoXSqrtX = mkBinOp("*", twoX, ctx.mathCall("sqrt", arg, elemTyp), elemTyp)
      let rule = mkPrefix("-", mkBinOp("/", g, twoXSqrtX, elemTyp))
      transposeExpr(ctx, arg, rule, elemTyp, env, path, into)
    else:
      adRefuse(e, "gpuCall",
        "call to '" & e.cName.ident() & "' is outside the differentiable subset")
  of gpuIndex:
    # Memory read: scatter-add the adjoint into the buffer's shadow adjoint.
    let t = ctx.getExprType(e)
    if t.isNil or t.kind notin DiffFloatKinds:
      return
    let root = indexRootParam(e.iArr)
    if root.isNil or root.symbol.isNil:
      adRefuse(e, "gpuIndex", "index root is not a buffer parameter")
    let dIdent = env.dIdents.getOrDefault(root.symbol.iSym, nil)
    if dIdent.isNil:
      adRefuse(e, "gpuIndex",
        "no shadow adjoint for buffer '" & root.symbol.name & "'")
    let idx = substituteLocals(ctx, e.iIndex, env, path)
    let gSub = substituteLocals(ctx, g, env, path)
    let lhs = mkIndex(dIdent, idx)
    into.add mkAssign(lhs, mkBinOp("+", mkIndex(dIdent, idx), gSub, elemTyp))
  of gpuTernary:
    let t = ctx.getExprType(e)
    if t.isNil or t.kind notin DiffFloatKinds:
      return
    let cond = substituteLocals(ctx, e.tCond, env, path)
    transposeExpr(ctx, e.tThen,
      GpuAst(kind: gpuTernary, tCond: cond, tThen: g, tElse: mkZeroLit(elemTyp)),
      elemTyp, env, path, into)
    transposeExpr(ctx, e.tElse,
      GpuAst(kind: gpuTernary, tCond: cond, tThen: mkZeroLit(elemTyp), tElse: g),
      elemTyp, env, path, into)
  of gpuConv:
    if e.convTo.kind notin DiffFloatKinds:
      adRefuse(e, "gpuConv",
        "conversion to " & pretty(e.convTo) &
        " of a differentiated value has no gradient")
    let innerT = ctx.getExprType(e.convExpr)
    if innerT.isNil or innerT.kind notin DiffFloatKinds:
      # rounding an operand through a bool or integer type silently
      # loses the gradient, no conversion direction repairs it
      adRefuse(e, "gpuConv",
        "conversion from " & pretty(innerT) &
        " through a differentiated value loses the gradient")
    let gInner = if innerT.kind == e.convTo.kind: g else: mkConv(innerT, g)
    transposeExpr(ctx, e.convExpr, gInner, elemTyp, env, path, into)
  of gpuMaterialize:
    transposeExpr(ctx, e.mExpr, g, elemTyp, env, path, into)
  else:
    refuseNodeKind(e)

# ═══════════════════════════════════════════════════════════════════════
# Statement transposition
# ═══════════════════════════════════════════════════════════════════════

proc transposeStmts(ctx: var GpuContext, stmts: seq[GpuAst], env: var AdEnv, path: seq[string]): seq[GpuAst] =
  ## Cotangent groups of a statement list, in forward order.
  ## The groups accumulate additively into zero-filled shadow buffers,
  ## so the group order does not change the sums.
  result = @[]
  for s in stmts:
    case s.kind
    of gpuAssign:
      if s.aLeft.kind == gpuIndex:
        let root = indexRootParam(s.aLeft.iArr)
        let dIdent = env.dIdents.getOrDefault(root.symbol.iSym, nil)
        let seedTyp = ctx.getExprType(s.aLeft)
        if dIdent.isNil or seedTyp.isNil:
          adRefuse(s, "gpuAssign", "no shadow adjoint for the written slot")
        let g = mkIndex(dIdent, substituteLocals(ctx, s.aLeft.iIndex, env, path))
        transposeExpr(ctx, s.aRight, g, seedTyp.clone(), env, path, result)
    of gpuFor:
      let savedAmbient = env.ambientVar
      let savedCount = env.ambientCount
      let savedScope = env.currentLoopScope
      env.ambientVar = s.fVar
      env.ambientCount = s.fEnd
      env.currentLoopScope = "for" & s.fVar.symbol.iSym
      let innerPath = path.scopePush("for" & s.fVar.symbol.iSym)
      let inner = transposeStmts(ctx, s.fBody.statements,
                                 env, innerPath)
      env.ambientVar = savedAmbient
      env.ambientCount = savedCount
      env.currentLoopScope = savedScope
      if inner.len > 0:
        var loop = GpuAst(kind: gpuFor,
                          fVar: s.fVar, fStart: s.fStart, fEnd: s.fEnd,
                          fStep: s.fStep, fRangeKind: s.fRangeKind)
        loop.fBody = GpuAst(kind: gpuBlock, statements: inner)
        result.add loop
    of gpuIf:
      let thenStmts = transposeStmts(ctx, s.ifThen.statements,
                                     env, path.scopePush("then"))
      var elifs: seq[tuple[cond, body: GpuAst]]
      for i in 0 ..< s.ifElifs.len:
        let el = s.ifElifs[i]
        let body = transposeStmts(ctx, el.body.statements,
                                  env, path.scopePush("elif" & $i))
        if body.len > 0:
          elifs.add (cond: el.cond, body: GpuAst(kind: gpuBlock, statements: body))
      var elseBody: GpuAst = GpuAst(kind: gpuDiscard)
      if s.ifElse.kind != gpuDiscard:
        let body = transposeStmts(ctx, s.ifElse.statements,
                                  env, path.scopePush("else"))
        if body.len > 0:
          elseBody = GpuAst(kind: gpuBlock, statements: body)
      if thenStmts.len > 0 or elifs.len > 0 or elseBody.kind != gpuDiscard:
        let mirrored = GpuAst(kind: gpuIf,
                              ifCond: substituteLocals(ctx, s.ifCond, env, path),
                              ifThen: GpuAst(kind: gpuBlock, statements: thenStmts),
                              ifIsExpr: false,
                              ifElifs: elifs,
                              ifElse: elseBody)
        result.add mirrored
    of gpuBlock:
      for emitted in transposeStmts(ctx, s.statements, env, path):
        result.add emitted
    of gpuDiscard, gpuComment, gpuConstexpr, gpuVar, gpuMaterialize:
      discard
    else:
      refuseNodeKind(s)

# ═══════════════════════════════════════════════════════════════════════
# Kernel assembly
# ═══════════════════════════════════════════════════════════════════════

proc collectFwdParamUses(n: GpuAst, paramISyms: HashSet[string], found: var HashSet[string]) =
  ## Collects the iSyms of forward parameters referenced by the backward body.
  ## Those parameters travel in the second block of the backward signature,
  ## after the d-block.
  if n.isNil:
    return
  if n.kind == gpuIdent and n.symbol != nil and n.symbol.iSym in paramISyms:
    found.incl n.symbol.iSym
    return
  for ch in n.items:
    collectFwdParamUses(ch, paramISyms, found)

func paramISyms(fn: GpuAst): HashSet[string] =
  ## iSym set of a kernel's parameters.
  result = initHashSet[string]()
  for p in fn.pParams:
    result.incl p.ident.symbol.iSym

proc makeReduceKernel(ctx: var GpuContext, fn: GpuAst, p: GpuParam, env: AdEnv): GpuAst =
  ## Generates the axis-reduce kernel of one scalar cotangent:
  ##
  ## - sums the per-index partials from the backward kernel's workspace
  ## - writes the sum into the scalar's 1-element gradient buffer
  ## - gradient name = the forward scalar param's NAME, type buffer(1)
  let elemTyp = p.typ.clone()
  let partialsTyp = mkBufferPtr(elemTyp)
  let outTyp = mkBufferPtr(elemTyp)
  let intTyp = ctx.getExprType(env.scalarLoopCount)
  if intTyp.isNil or intTyp.kind notin {gtInt32, gtInt64, gtUint32, gtUint64, gtSize_t}:
    adRefuse(fn, "gpuFor",
      "cannot derive the reduce-kernel trip count from the index loop bound")

  let partialsSym = newSymbol("d_" & p.ident.ident() & "_partials",
                              iSym = "d_" & p.ident.symbol.iSym & "_partials",
                              typ = partialsTyp, symKind = gsDeviceKernelParam)
  let outSym = newSymbol("d_" & p.ident.ident(),
                         iSym = "d_" & p.ident.symbol.iSym,
                         typ = outTyp, symKind = gsDeviceKernelParam)
  let nSym = newSymbol("n", iSym = p.ident.symbol.iSym & "_reduce_n",
                       typ = intTyp, symKind = gsDeviceKernelParam)
  let iSym = newSymbol("i", iSym = p.ident.symbol.iSym & "_reduce_i",
                       typ = intTyp, symKind = gsLocal)
  let accSym = newSymbol("acc", iSym = p.ident.symbol.iSym & "_reduce_acc",
                         typ = elemTyp, symKind = gsLocal)
  let partialsIdent = GpuAst(kind: gpuIdent, symbol: partialsSym)
  let outIdent = GpuAst(kind: gpuIdent, symbol: outSym)
  let nIdent = GpuAst(kind: gpuIdent, symbol: nSym)
  let iIdent = GpuAst(kind: gpuIdent, symbol: iSym)
  let accIdent = GpuAst(kind: gpuIdent, symbol: accSym)

  let loopBody = GpuAst(kind: gpuBlock, statements: @[
    mkAssign(accIdent, mkBinOp("+", accIdent,
                               mkIndex(partialsIdent, iIdent), elemTyp))
  ])
  let loop = GpuAst(kind: gpuFor,
                    fVar: iIdent,
                    fStart: GpuAst(kind: gpuLit, lValue: "0", lType: intTyp),
                    fEnd: nIdent,
                    fStep: GpuAst(kind: gpuLit, lValue: "1", lType: intTyp),
                    fBody: loopBody,
                    fRangeKind: rkExclusive)

  result = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent,
                                symbol: newSymbol(fn.pName.ident() & "_bwd_d_" &
                                                  p.ident.ident() & "_reduce",
                                                  iSym = fn.pName.symbol.iSym &
                                                  "_bwd_d_" & p.ident.symbol.iSym &
                                                  "_reduce",
                                                  symKind = gsProc)),
                  pRetType: GpuType(kind: gtVoid),
                  pParams: @[
                    GpuParam(ident: partialsIdent, typ: partialsTyp,
                             addressSpace: asDevice, passByRef: false, flavor: kIn),
                    GpuParam(ident: nIdent, typ: intTyp,
                             addressSpace: asRMEM, passByRef: false, flavor: kIn),
                    GpuParam(ident: outIdent, typ: outTyp,
                             addressSpace: asDevice, passByRef: false, flavor: kIn)
                  ],
                  pBody: GpuAst(kind: gpuBlock, statements: @[
                    GpuAst(kind: gpuVar, vName: accIdent, vType: elemTyp,
                           vInit: mkZeroLit(elemTyp), vRequiresMemcpy: false,
                           vMutable: true, addressSpace: asRMEM),
                    loop,
                    mkAssign(mkIndex(outIdent,
                                     GpuAst(kind: gpuLit, lValue: "0",
                                            lType: intTyp)),
                             accIdent)
                  ]),
                  pAttributes: {attGlobal})

proc transposeKernel(ctx: var GpuContext, fn: GpuAst) =
  ## Transposes one global kernel into its backward kernel plus one
  ## axis-reduce kernel per scalar cotangent. The forward kernel stays
  ## as it is.
  var env = AdEnv()
  for p in fn.pParams:
    if p.typ.kind == gtPtr:
      env.bufferParams[p.ident.symbol.iSym] = p
    elif p.typ.kind in DiffFloatKinds:
      env.scalarParams[p.ident.symbol.iSym] = p

  validateStmts(ctx, fn.pBody.statements, env, @[], 0)

  if env.written.len == 0:
    adRefuse(fn, "gpuProc", "kernel writes no buffer, nothing to differentiate")
  for w in env.written:
    if w in env.readBufs:
      adRefuse(fn, "gpuProc",
        "kernel reads buffer '" & env.bufferParams[w].ident.ident() &
        "' it also writes; the differentiable subset is out-of-place")

  # Shadow-adjoint idents: d_<buffer> for every buffer the kernel touches.
  for iSym, p in env.bufferParams:
    if iSym in env.written or iSym in env.readBufs:
      let sym = newSymbol("d_" & p.ident.ident(),
                          iSym = "d_" & p.ident.symbol.iSym,
                          typ = p.typ.clone(), symKind = gsDeviceKernelParam)
      env.dIdents[iSym] = GpuAst(kind: gpuIdent, symbol: sym)

  let groups = transposeStmts(ctx, fn.pBody.statements, env, @[])

  # Backward signature: the d-block first, gradients of the forward params
  # in forward order, then needed forward params, then reduce workspaces.
  var needed: HashSet[string]
  collectFwdParamUses(GpuAst(kind: gpuBlock, statements: groups),
                      paramISyms(fn), needed)

  var params: seq[GpuParam]
  for p in fn.pParams:
    if p.typ.kind == gtPtr and p.ident.symbol.iSym in env.dIdents:
      params.add GpuParam(ident: env.dIdents[p.ident.symbol.iSym],
                          typ: p.typ.clone(), addressSpace: p.addressSpace,
                          passByRef: p.passByRef, flavor: kIn)
  for p in fn.pParams:
    if p.ident.symbol.iSym in needed:
      params.add p
  var reduceKernels: seq[GpuAst]
  for p in fn.pParams:
    if p.typ.kind in DiffFloatKinds and p.ident.symbol.iSym in env.scalarHits:
      let partialsIdent = env.partials[p.ident.symbol.iSym]
      params.add GpuParam(ident: partialsIdent,
                          typ: partialsIdent.symbol.typ.clone(),
                          addressSpace: asDevice, passByRef: false, flavor: kIn)
      reduceKernels.add ctx.makeReduceKernel(fn, p, env)

  let bwd = GpuAst(kind: gpuProc,
                   pName: GpuAst(kind: gpuIdent,
                                 symbol: newSymbol(fn.pName.ident() & "_bwd",
                                                   iSym = fn.pName.symbol.iSym &
                                                   "_bwd",
                                                   symKind = gsProc)),
                   pRetType: GpuType(kind: gtVoid),
                   pParams: params,
                   pBody: GpuAst(kind: gpuBlock, statements: groups),
                   pAttributes: fn.pAttributes,
                   pWorkgroupSize: fn.pWorkgroupSize)
  ctx.allFnTab[bwd.pName] = bwd
  ctx.fnTab[bwd.pName] = bwd
  for rk in reduceKernels:
    ctx.allFnTab[rk.pName] = rk
    ctx.fnTab[rk.pName] = rk

proc autodiffTranspose(ctx: var GpuContext) =
  ## Pass entry: transposes every global kernel in the context.
  var kernels: seq[GpuAst]
  for fn in ctx.allFnTab.values():
    if fn.kind == gpuProc and attGlobal in fn.pAttributes and
        not fn.pName.ident().endsWith("_bwd"):
      kernels.add fn
  for fn in kernels:
    ctx.transposeKernel(fn)
