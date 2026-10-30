# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cirv2 normal-form verifier: the printer-side Leng contract.
##
## Every pass wrapper runs `verify` (or `verifyAfter`) over its output.
## A violation is a compile-time rejection through `macros.error`.
## A rejection message names the construct, its node path, and the detail.
## `verifyMessage` runs the same walk purely and returns the message.
##
## | Invariant    | Contract                                                                                 |
## | ------------ | ---------------------------------------------------------------------------------------- |
## | Types        | every operator, literal, cast, and constructor names its operand or result type          |
## | Types        | the two `asgn` sides are same-lattice types                                              |
## | Types        | the `asgn` lhs is an lvalue                                                              |
## | Types        | a `!=` against the zero u32 literal compares a bool (the WGSL bool launch ABI)           |
## | Control flow | `jmp` is forward-only and scoped (target `lab` textually later, same or enclosing scope) |
## | Control flow | every `lab` is proc-unique and targeted                                                  |
## | Control flow | `ite` conditions are bool                                                                |
## | Control flow | `case` ranges are disjoint literal ranges over an int/uint selector                      |
## | Calls        | `eCall` appears only as the whole RHS of `asgn`                                          |
## | Calls        | `sCall` appears only as a void statement                                                 |
## | Calls        | a call never appears under `and`/`or` or in any other operand position                   |
## | Returns      | `ret` matches the proc return type                                                       |
## | Assumptions  | `assume` conditions are comparisons or and/or trees over comparisons                     |
## | Structs      | struct `dot` inheritance is 0                                                            |
import cirv2_lang
import std/[macros, sets, strutils]

type
  VerifyCtx = object
    ## Walk state for one proc verification.
    fnName: string
    retType: CirType
    err: string
    ## First rejection message body, "" while the walk is clean.
    ## Later violations are dropped: the walk unwinds through the err guards.
    seenLabels: HashSet[string]
    ## Labels seen so far, a jmp to one of these is backward.
    declaredLabs: HashSet[string]
    targetedLabs: HashSet[string]
    pendingJmps: seq[PendingJmp]
    nextScopeId: int

  PendingJmp = object
    name: string
    chain: seq[int]
    path: string

  ScopeChain = seq[int]
  ## Chain of open scope ids from the proc body down to the current statement list.
  ## A lab at chain `L` is reachable from a jmp with chain `J` iff `L` is a prefix of `J` (same or enclosing scope).

proc getConstructName(k: CirExprKind | CirStmtKind): string =
  ## Leng vocabulary name of a node kind (`eAdd` -> `add`, `sIte` -> `ite`).
  ## Every rejection message leads with this name.
  toLowerAscii($k)[1 ..^ 1]

proc fail(ctx: var VerifyCtx, path: string, construct: string, detail: string) =
  if ctx.err.len == 0:
    ctx.err = construct & " at " & ctx.fnName & "/" & path & ": " & detail

proc checkOperand(ctx: var VerifyCtx, e: CirExpr, path: string, side: string): CirType =
  ## Returns the type of a child expression, rejecting when the shape does
  ## not determine one.
  result = getExprType(e)
  if result.isNil:
    ctx.fail(path, getConstructName(e.kind), side & " has no derivable type")

proc spellTypeDbg(t: CirType): string =
  ## Type text for a rejection message: the struct name when present, otherwise the lattice kind.
  if t.isNil:
    return "nil"
  if t.kind == ctStruct:
    return t.name
  if t.kind in {ctPtr, ctAptr}:
    return $t.kind & "(" & $t.space & " " & spellTypeDbg(t.pElem) & ")"
  return $t.kind

proc sameOrFails(ctx: var VerifyCtx, a, b: CirType, path: string, construct: string, detail: string) =
  if not sameType(a, b):
    ctx.fail(path, construct,
      detail & ": got " & spellTypeDbg(a) & ", expected " & spellTypeDbg(b))

proc callArgReSpace(ctx: var VerifyCtx, a, b: CirType, path: string) =
  ## Call-argument check under v1's birth-space rule: a pointer argument
  ## may hold another address space than the formal states.
  ## The call site re-spaces the argument to the formal's space.
  ## The pointee type must match exactly.
  if sameType(a, b):
    return
  if a.kind == b.kind and a.kind in {ctPtr, ctAptr} and sameType(a.pElem, b.pElem):
    return
  ctx.fail(path, "call",
    "argument type differs from the parameter type: got " &
    spellTypeDbg(a) & ", expected " & spellTypeDbg(b))

proc retReSpace(ctx: var VerifyCtx, a, b: CirType, path: string) =
  ## Return-value check under v1's birth-space rule: a pointer return may
  ## carry another address space than the proc return type states.
  ## The call site re-spaces to the declared space.
  ## The pointee type must match exactly.
  if sameType(a, b):
    return
  if a.kind == b.kind and a.kind in {ctPtr, ctAptr} and sameType(a.pElem, b.pElem):
    return
  ctx.fail(path, "ret",
    "return type differs from the proc return type: got " &
    spellTypeDbg(a) & ", expected " & spellTypeDbg(b))

proc litFitsBits(ctx: var VerifyCtx, v: int64, t: CirType, path: string) =
  ## Fails when an int64 value does not fit the named type's bit width.
  if t.bits <= 0:
    return
  if t.kind == ctInt:
    if t.bits >= 64:
      return
    let lim = 1'i64 shl (t.bits - 1)
    if v < -lim or v > lim - 1:
      ctx.fail(path, "int literal", "value " & $v & " does not fit (i " & $t.bits & ")")
  elif t.kind == ctUint:
    if t.bits >= 63:
      return
    let lim = 1'i64 shl t.bits
    if v < 0 or v >= lim:
      ctx.fail(path, "int literal", "value " & $v & " does not fit (u " & $t.bits & ")")

proc verifyExpr(ctx: var VerifyCtx, e: CirExpr, path: string, allowCall: bool): CirType =
  ## Validates one expression node and its subtree.
  ## `allowCall` is true only at the node the caller names.
  ## A call is legal as the whole RHS of an `asgn` and stays nested
  ## inside no other expression.
  if ctx.err.len > 0:
    return nil
  if e.isNil:
    ctx.fail(path, "expr", "missing expression")
    return nil
  case e.kind
  of eSym:
    if e.typ.isNil:
      ctx.fail(path, "sym", "symbol without type")
      return nil
    result = e.typ
  of eIntLit:
    if e.litType.isNil or e.litType.kind notin {ctInt, ctUint}:
      ctx.fail(path, "int literal", "literal type must be int or uint")
      return nil
    litFitsBits(ctx, e.value, e.litType, path)
    if ctx.err.len > 0:
      return nil
    result = e.litType
  of eFloatLit:
    if e.fLitType.isNil or e.fLitType.kind != ctFloat:
      ctx.fail(path, "float literal", "literal type must be float")
      return nil
    result = e.fLitType
  of eNil:
    if e.nilType.isNil or e.nilType.kind notin {ctPtr, ctAptr}:
      ctx.fail(path, "nil", "nil needs a spelled pointer type")
      return nil
    result = e.nilType
  of eTrue, eFalse:
    result = CirType(kind: ctBool)
  of eDeref:
    let t = checkOperand(ctx, e.derefOperand, path & ".operand", "deref operand")
    if ctx.err.len > 0:
      return nil
    if t.kind != ctPtr:
      ctx.fail(path, "deref", "operand is not a ptr")
      return nil
    result = t.pElem
  of eAt:
    let base = checkOperand(ctx, e.atBase, path & ".base", "at base")
    if ctx.err.len > 0:
      return nil
    if base.kind != ctArray:
      ctx.fail(path, "at", "base is not an array")
      return nil
    let idx = checkOperand(ctx, e.atIndex, path & ".index", "at index")
    if ctx.err.len > 0:
      return nil
    if idx.kind notin {ctInt, ctUint}:
      ctx.fail(path, "at", "index is not an integer")
      return nil
    result = base.aElem
  of ePat:
    let base = checkOperand(ctx, e.patBase, path & ".base", "pat base")
    if ctx.err.len > 0:
      return nil
    if base.kind != ctAptr:
      ctx.fail(path, "pat", "base is not an aptr")
      return nil
    let idx = checkOperand(ctx, e.patIndex, path & ".index", "pat index")
    if ctx.err.len > 0:
      return nil
    if idx.kind notin {ctInt, ctUint}:
      ctx.fail(path, "pat", "index is not an integer")
      return nil
    result = base.pElem
  of eDot:
    if e.inherit != 0:
      ctx.fail(path, "dot", "inheritance number must be 0: CIRv2 structs carry no parent chain")
      return nil
    let t = checkOperand(ctx, e.dotBase, path & ".base", "dot base")
    if ctx.err.len > 0:
      return nil
    if t.kind != ctStruct:
      ctx.fail(path, "dot", "base is not a struct")
      return nil
    var found = false
    for f in t.fields:
      if f.name == e.fieldName:
        result = f.typ
        found = true
    if not found:
      ctx.fail(path, "dot", "struct " & t.name & " has no field " & e.fieldName)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    if e.opType.isNil:
      ctx.fail(path, getConstructName(e.kind), "operator does not spell its type")
      return nil
    let lt = checkOperand(ctx, e.left, path & ".left", "left operand")
    if ctx.err.len > 0:
      return nil
    let rt = checkOperand(ctx, e.right, path & ".right", "right operand")
    if ctx.err.len > 0:
      return nil
    sameOrFails(ctx, lt, e.opType, path, getConstructName(e.kind),
      "left operand type differs from the spelled type")
    sameOrFails(ctx, rt, e.opType, path, getConstructName(e.kind),
      "right operand type differs from the spelled type")
    discard ctx.verifyExpr(e.left, path & ".left", false)
    discard ctx.verifyExpr(e.right, path & ".right", false)
    result = e.opType
  of eBitNot, eNeg:
    if e.unType.isNil:
      ctx.fail(path, getConstructName(e.kind), "operator does not spell its type")
      return nil
    let t = checkOperand(ctx, e.unOperand, path & ".operand", "operand")
    if ctx.err.len > 0:
      return nil
    sameOrFails(ctx, t, e.unType, path, getConstructName(e.kind),
      "operand type differs from the spelled type")
    discard ctx.verifyExpr(e.unOperand, path & ".operand", false)
    result = e.unType
  of eEq, eNeq, eLt, eLe:
    if e.cmpType.isNil:
      ctx.fail(path, getConstructName(e.kind), "comparison does not spell its operand type")
      return nil
    let at = checkOperand(ctx, e.a, path & ".a", "left operand")
    if ctx.err.len > 0:
      return nil
    let bt = checkOperand(ctx, e.b, path & ".b", "right operand")
    if ctx.err.len > 0:
      return nil
    sameOrFails(ctx, at, e.cmpType, path, getConstructName(e.kind),
      "left operand type differs from the spelled operand type")
    if e.kind == eNeq and at.kind == ctBool and e.b.kind == eIntLit and
        e.b.value == 0 and bt.kind == ctUint and bt.bits == 32:
      # ABI zero test of a bool value: the launch ABI stores a bool kernel
      # parameter as u32, and the WGSL lowering pass emits the read as `flag != 0u`
      discard
    else:
      sameOrFails(ctx, bt, e.cmpType, path, getConstructName(e.kind),
        "right operand type differs from the spelled operand type")
    discard ctx.verifyExpr(e.a, path & ".a", false)
    discard ctx.verifyExpr(e.b, path & ".b", false)
    result = CirType(kind: ctBool)
  of eNot:
    let t = checkOperand(ctx, e.notOperand, path & ".operand", "not operand")
    if ctx.err.len > 0:
      return nil
    if t.kind != ctBool:
      ctx.fail(path, "not", "operand is not bool")
      return nil
    discard ctx.verifyExpr(e.notOperand, path & ".operand", false)
    result = CirType(kind: ctBool)
  of eAnd, eOr:
    let lt = checkOperand(ctx, e.andLeft, path & ".left", "bool operand")
    if ctx.err.len > 0:
      return nil
    let rt = checkOperand(ctx, e.andRight, path & ".right", "bool operand")
    if ctx.err.len > 0:
      return nil
    if lt.kind != ctBool or rt.kind != ctBool:
      ctx.fail(path, getConstructName(e.kind), "operands are not bool")
      return nil
    # Operands are verified with calls banned: a call under a pure bool
    # operator is a side effect in expression position.
    discard ctx.verifyExpr(e.andLeft, path & ".left", false)
    discard ctx.verifyExpr(e.andRight, path & ".right", false)
    result = CirType(kind: ctBool)
  of eCast, eConv:
    if e.castType.isNil:
      ctx.fail(path, getConstructName(e.kind), "cast does not spell its result type")
      return nil
    discard checkOperand(ctx, e.castOperand, path & ".operand", "cast operand")
    discard ctx.verifyExpr(e.castOperand, path & ".operand", false)
    result = e.castType
  of eAddr:
    if e.addrType.isNil or e.addrType.kind != ctPtr:
      ctx.fail(path, "addr", "does not spell a ptr result type")
      return nil
    if not isLvalue(e.addrOperand):
      ctx.fail(path, "addr", "operand is not an lvalue")
      return nil
    discard checkOperand(ctx, e.addrOperand, path & ".operand", "addr operand")
    discard ctx.verifyExpr(e.addrOperand, path & ".operand", false)
    result = e.addrType
  of eSizeof:
    if e.ofType.isNil:
      ctx.fail(path, "sizeof", "missing operand type")
      return nil
    result = CirType(kind: ctUint, bits: -1)
  of eOconstr:
    if e.ocType.isNil or e.ocType.kind != ctStruct:
      ctx.fail(path, "oconstr", "does not spell a struct type")
      return nil
    for i, kvField in e.kvFields:
      var found = false
      for f in e.ocType.fields:
        if f.name == kvField.name:
          found = true
          sameOrFails(ctx, getExprType(kvField.value), f.typ,
            path & ".kv[" & $i & "]", "oconstr",
            "field " & kvField.name & " value type differs from the field type")
      if not found:
        ctx.fail(path, "oconstr",
          "struct " & e.ocType.name & " has no field " & kvField.name)
        return nil
      discard ctx.verifyExpr(kvField.value, path & ".kv[" & $i & "]", false)
    result = e.ocType
  of eAconstr:
    if e.acType.isNil or e.acType.kind != ctArray:
      ctx.fail(path, "aconstr", "does not spell an array type")
      return nil
    if e.elems.len != e.acType.aN:
      ctx.fail(path, "aconstr",
        "element count " & $e.elems.len & " differs from array length " & $e.acType.aN)
      return nil
    for i, el in e.elems:
      let t = checkOperand(ctx, el, path & ".elem[" & $i & "]", "array element")
      if ctx.err.len > 0:
        return nil
      sameOrFails(ctx, t, e.acType.aElem, path & ".elem[" & $i & "]",
        "aconstr", "element type differs from the array element type")
      discard ctx.verifyExpr(el, path & ".elem[" & $i & "]", false)
    result = e.acType
  of eCall:
    if not allowCall:
      ctx.fail(path, "call", "call in operand position")
      return nil
    let calleeType = checkOperand(ctx, e.callee, path & ".callee", "callee")
    if ctx.err.len > 0:
      return nil
    if calleeType.kind != ctProc:
      ctx.fail(path, "call", "callee is not a proc")
      return nil
    if calleeType.ret.kind == ctVoid:
      ctx.fail(path, "call", "void call must be a statement, not a value")
      return nil
    if e.args.len != calleeType.params.len:
      ctx.fail(path, "call",
        "argument count " & $e.args.len & " differs from parameter count " &
        $calleeType.params.len)
      return nil
    for i, arg in e.args:
      let t = checkOperand(ctx, arg, path & ".arg[" & $i & "]", "call argument")
      if ctx.err.len > 0:
        return nil
      callArgReSpace(ctx, t, calleeType.params[i], path & ".arg[" & $i & "]")
      discard ctx.verifyExpr(arg, path & ".arg[" & $i & "]", false)
    result = calleeType.ret

proc verifyCallArgs(ctx: var VerifyCtx, callee: CirExpr, args: seq[CirExpr], path: string) =
  ## Shared shape check for statement-position calls.
  if ctx.err.len > 0:
    return
  let calleeType = checkOperand(ctx, callee, path & ".callee", "callee")
  if ctx.err.len > 0:
    return
  if calleeType.kind != ctProc:
    ctx.fail(path, "call", "callee is not a proc")
    return
  if calleeType.ret.kind != ctVoid:
    ctx.fail(path, "call",
      "non-void call must bind to a destination (asgn RHS), not be a statement")
    return
  if args.len != calleeType.params.len:
    ctx.fail(path, "call",
      "argument count " & $args.len & " differs from parameter count " &
      $calleeType.params.len)
    return
  for i, arg in args:
    let t = checkOperand(ctx, arg, path & ".arg[" & $i & "]", "call argument")
    if ctx.err.len > 0:
      return
    callArgReSpace(ctx, t, calleeType.params[i], path & ".arg[" & $i & "]")
    discard ctx.verifyExpr(arg, path & ".arg[" & $i & "]", false)

proc isComparisonShaped(e: CirExpr): bool =
  ## An `assume` condition: a comparison, or an and/or tree over comparisons.
  case e.kind
  of eEq, eNeq, eLt, eLe:
    result = true
  of eAnd, eOr:
    result = isComparisonShaped(e.andLeft) and isComparisonShaped(e.andRight)
  else:
    result = false

proc verifyStmtList(ctx: var VerifyCtx, body: seq[CirStmt], path: string, chain: ScopeChain)

proc freshScope(ctx: var VerifyCtx): int =
  ctx.nextScopeId.inc
  ctx.nextScopeId

proc verifyBranchList(ctx: var VerifyCtx, body: seq[CirStmt], path: string, chain: ScopeChain) =
  ## A branch body (ite variant, loop body, case variant, scope) is a fresh scope.
  if ctx.err.len > 0:
    return
  var inner = chain
  inner.add(ctx.freshScope)
  ctx.verifyStmtList(body, path, inner)

proc verifyStmt(ctx: var VerifyCtx, s: CirStmt, path: string, chain: ScopeChain) =
  if ctx.err.len > 0:
    return
  if s.isNil:
    ctx.fail(path, "stmt", "missing statement")
    return
  case s.kind
  of sVar:
    if s.declType.isNil or s.declType.kind == ctVoid:
      ctx.fail(path, "var", "declaration needs a non-void type")
      return
    if s.init != nil:
      let it = checkOperand(ctx, s.init, path & ".init", "initializer")
      if ctx.err.len > 0:
        return
      sameOrFails(ctx, it, s.declType, path & ".init", "var",
        "initializer type differs from the declared type: init " & $(if it.isNil: "nil" else: $it.kind & "/" & (if it.kind == ctStruct: it.name else: "")) & " vs decl " & $(if s.declType.kind == ctStruct: s.declType.name else: $s.declType.kind))
      # Calls bind to their destination via asgn. A declaration is not a call.
      discard ctx.verifyExpr(s.init, path & ".init", false)
  of sConst:
    if s.declType.isNil or s.declType.kind == ctVoid:
      ctx.fail(path, "const", "declaration needs a non-void type")
      return
    if s.init == nil:
      ctx.fail(path, "const", "constant without a value")
      return
    let it = checkOperand(ctx, s.init, path & ".init", "value")
    if ctx.err.len > 0:
      return
    sameOrFails(ctx, it, s.declType, path & ".init", "const",
      "value type differs from the declared type")
    discard ctx.verifyExpr(s.init, path & ".init", false)
  of sAsgn:
    if not isLvalue(s.lhs):
      ctx.fail(path, "asgn", "left-hand side is not an lvalue")
      return
    let lhsType = checkOperand(ctx, s.lhs, path & ".lhs", "left-hand side")
    if ctx.err.len > 0:
      return
    if s.rhs.kind == eCall:
      let rhsType = ctx.verifyExpr(s.rhs, path & ".rhs", true)
      if ctx.err.len > 0:
        return
      sameOrFails(ctx, rhsType, lhsType, path & ".rhs", "asgn",
        "call result type differs from the destination type")
    else:
      let rhsType = checkOperand(ctx, s.rhs, path & ".rhs", "right-hand side")
      if ctx.err.len > 0:
        return
      sameOrFails(ctx, rhsType, lhsType, path & ".rhs", "asgn",
        "right-hand side type differs from the destination type")
      discard ctx.verifyExpr(s.rhs, path & ".rhs", false)
  of sIte:
    let ct = checkOperand(ctx, s.cond, path & ".cond", "ite condition")
    if ctx.err.len > 0:
      return
    if ct.kind != ctBool:
      ctx.fail(path, "ite", "condition is not bool")
      return
    discard ctx.verifyExpr(s.cond, path & ".cond", false)
    ctx.verifyBranchList(s.thenBody, path & ".then", chain)
    ctx.verifyBranchList(s.elseBody, path & ".else", chain)
  of sLoop:
    # Shape-enforced: a loop has no condition slot, guards live in the body.
    ctx.verifyBranchList(s.loopBody, path & ".body", chain)
  of sCase:
    let st = checkOperand(ctx, s.selector, path & ".selector", "case selector")
    if ctx.err.len > 0:
      return
    if st.kind notin {ctInt, ctUint}:
      ctx.fail(path, "case", "selector is not an integer type")
      return
    if st.kind == ctUint and st.bits >= 63:
      ctx.fail(path, "case",
        "selector wider than 63 unsigned bits: range literals are int64")
      return
    var allRanges: seq[CirRange] = @[]
    for bi, br in s.branches:
      if br.ranges.len == 0:
        ctx.fail(path & ".of[" & $bi & "]", "case", "branch without ranges")
        return
      for ri, r in br.ranges:
        if r.lo > r.hi:
          ctx.fail(path & ".of[" & $bi & "].range[" & $ri & "]", "case",
            "empty range " & $r.lo & ".." & $r.hi)
          return
        if st.kind == ctUint and r.lo < 0:
          ctx.fail(path & ".of[" & $bi & "].range[" & $ri & "]", "case",
            "negative range bound over an unsigned selector")
          return
        litFitsBits(ctx, r.lo, st, path & ".of[" & $bi & "]")
        litFitsBits(ctx, r.hi, st, path & ".of[" & $bi & "]")
        allRanges.add(r)
      ctx.verifyBranchList(br.body, path & ".of[" & $bi & "]", chain)
    for i in 0 ..< allRanges.len:
      for j in i + 1 ..< allRanges.len:
        if allRanges[i].lo <= allRanges[j].hi and
           allRanges[j].lo <= allRanges[i].hi:
          ctx.fail(path, "case",
            "overlapping ranges " & $allRanges[i].lo & ".." & $allRanges[i].hi &
            " and " & $allRanges[j].lo & ".." & $allRanges[j].hi)
    ctx.verifyBranchList(s.caseElse, path & ".else", chain)
  of sLab:
    if s.labName.len == 0:
      ctx.fail(path, "lab", "empty label name")
      return
    if s.labName in ctx.declaredLabs:
      ctx.fail(path, "lab", "duplicate label " & s.labName)
      return
    ctx.declaredLabs.incl(s.labName)
    ctx.seenLabels.incl(s.labName)
    # Resolve pending forward jmps: each must target this lab from the same
    # or an enclosing scope (its chain is a prefix of the jmp's chain).
    var i = 0
    while i < ctx.pendingJmps.len:
      if ctx.pendingJmps[i].name == s.labName:
        let j = ctx.pendingJmps[i]
        let ok = j.chain.len >= chain.len and chain == j.chain[0 ..< chain.len]
        if not ok:
          ctx.fail(j.path, "jmp",
            "target lab " & s.labName & " lies in a scope the jmp would enter")
          return
        ctx.pendingJmps.delete(i)
      else:
        i.inc
  of sJmp:
    if s.jmpTarget.len == 0:
      ctx.fail(path, "jmp", "empty jump target")
      return
    if s.jmpTarget in ctx.seenLabels:
      ctx.fail(path, "jmp", "backward jump to " & s.jmpTarget)
      return
    ctx.targetedLabs.incl(s.jmpTarget)
    ctx.pendingJmps.add(PendingJmp(name: s.jmpTarget, chain: chain, path: path))
  of sScope:
    ctx.verifyBranchList(s.scopeBody, path & ".body", chain)
  of sAssume:
    if not isComparisonShaped(s.assumeCond):
      ctx.fail(path, "assume",
        "condition is not a comparison or a conjunction over comparisons")
      return
    discard ctx.verifyExpr(s.assumeCond, path & ".cond", false)
  of sRet:
    if s.retVal == nil:
      if ctx.retType.kind != ctVoid:
        ctx.fail(path, "ret", "empty return from a non-void proc")
      return
    if ctx.retType.kind == ctVoid:
      ctx.fail(path, "ret", "value returned from a void proc")
      return
    let t = checkOperand(ctx, s.retVal, path & ".value", "return value")
    if ctx.err.len > 0:
      return
    retReSpace(ctx, t, ctx.retType, path & ".value")
    discard ctx.verifyExpr(s.retVal, path & ".value", false)
  of sDiscard:
    if s.discardVal != nil:
      discard checkOperand(ctx, s.discardVal, path & ".value", "discarded value")
      discard ctx.verifyExpr(s.discardVal, path & ".value", false)
  of sCall:
    ctx.verifyCallArgs(s.callCallee, s.callArgs, path)
  of sAsm:
    if s.asmParts.len != s.asmOperands.len + 1:
      ctx.fail(path, "asm", "template pieces and operands do not interleave")
      return
    for i, op in s.asmOperands:
      if op.kind != eSym or op.typ == nil:
        ctx.fail(path, "asm", "operand " & $i & " is not a typed symbol")
        return
  of sEmit:
    if s.emitParts.len != s.emitOperands.len + 1:
      ctx.fail(path, "emit", "template pieces and operands do not interleave")
      return
    for i, op in s.emitOperands:
      if op.kind != eSym or op.typ == nil:
        ctx.fail(path, "emit", "operand " & $i & " is not a typed symbol")
        return

proc verifyStmtList(ctx: var VerifyCtx, body: seq[CirStmt], path: string, chain: ScopeChain) =
  for i, s in body:
    if ctx.err.len > 0:
      return
    ctx.verifyStmt(s, path & "[" & $i & "]", chain)

proc checkProc(fn: CirProc): string =
  ## Pure contract walk over one proc: "" when clean, else the rejection
  ## message body ("<construct> at <proc>/<path>: <detail>").
  if fn.isNil:
    return "missing proc"
  var ctx = VerifyCtx(fnName: fn.name, retType: fn.ret)
  var chain: ScopeChain = @[ctx.freshScope]
  for pi, p in fn.params:
    if p.typ.isNil or p.typ.kind == ctVoid:
      ctx.fail("param[" & $pi & "]", "param", "parameter needs a non-void type")
      return ctx.err
  if fn.ret.isNil:
    return fn.name & ": proc without a spelled return type"
  ctx.verifyStmtList(fn.body, "body", chain)
  if ctx.err.len > 0:
    return ctx.err
  if ctx.pendingJmps.len > 0:
    let j = ctx.pendingJmps[0]
    return "jmp at " & j.path & ": unknown label " & j.name
  for labName in ctx.declaredLabs:
    if labName notin ctx.targetedLabs:
      return "lab at " & fn.name & ": label " & labName & " is never targeted"

proc verifyMessage*(fn: CirProc, passName = ""): string =
  ## Pure rejection check: returns "" when `fn` verifies clean, else the rejection message.
  ## The message reads "cirv2 verify: <construct> at <proc>/<path>: <detail>".
  ## With `passName` the message gains the "[pass <name>] " prefix, matching the `verifyAfter` rejection message.
  ##
  ## Example:
  ##   doAssert verifyMessage(badProc()).contains("ite at f/body[0]")
  let msg = checkProc(fn)
  if msg.len == 0:
    return ""
  if passName.len > 0:
    return "[pass " & passName & "] cirv2 verify: " & msg
  result = "cirv2 verify: " & msg

proc verify*(fn: CirProc) =
  ## Verifies one proc against the normal-form contract.
  ## The first violation rejects through `error` with the construct, node path, and detail.
  let msg = checkProc(fn)
  if msg.len > 0:
    error("cirv2 verify: " & msg)

proc verifyAfter*(passName: string, fn: CirProc) =
  ## Pass-wrapper sugar: verifies `fn` and prefixes every rejection with the "[pass <name>] " prefix.
  ## The violation then surfaces at the pass that created it.
  ##
  ## Examples:
  ##   verifyAfter("norm_ssa", fn)
  let msg = verifyMessage(fn, passName)
  if msg.len > 0:
    error(msg)

proc verifyMessage*(m: CirModule): string =
  ## Pure rejection check for a module: returns "" when `m` verifies clean, else the rejection message.
  ## The message names the module and the offending proc index, matching the `verify` rejection message.
  if m.isNil:
    return "cirv2 verify: missing module"
  for i, t in m.types:
    if t.isNil:
      return "cirv2 verify: " & m.name & "/type[" & $i & "]: missing type"
    if t.kind == ctStruct and t.name.len == 0:
      return "cirv2 verify: " & m.name & "/type[" & $i & "]: struct without a name"
  for i, fn in m.procs:
    let msg = checkProc(fn)
    if msg.len > 0:
      return "cirv2 verify: " & m.name & "/proc[" & $i & "]: " & msg
  for i, g in m.globals:
    # program-scope declarations hoisted by a lowering pass: the walk checks
    # var/const shape without a proc context, so label and return bookkeeping
    # does not apply
    var gctx = VerifyCtx(fnName: m.name & "/globals[" & $i & "]",
      retType: CirType(kind: ctVoid))
    let chain: ScopeChain = @[gctx.freshScope]
    gctx.verifyStmt(g, "body", chain)
    if gctx.err.len > 0:
      return "cirv2 verify: " & gctx.err
  return ""

proc verify*(m: CirModule) =
  ## Verifies every proc of a module. A violation rejects through `error`
  ## with the module name and proc index in the message.
  let msg = verifyMessage(m)
  if msg.len > 0:
    error(msg)
