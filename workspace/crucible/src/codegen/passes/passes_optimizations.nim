# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Optimization stage over CIRv2: arithmetic identity folding, single-use temp folding,
## and dead-local elimination.
##
## - identity folding: a `mul`/`div` by the literal 1 or an integer `add`/`sub`
##   with the literal 0 collapses to the surviving operand, one bottom-up walk
## - single-use temp folding: a temp assigned once and read once folds into
##   its only reader and the store deletes
## - dead-local elimination: a local that nothing reads afterwards is deleted,
##   its type never reaches the printer
## - folding runs first so its eliminations cascade into the dead-local pass

import std/[sequtils, sets, tables]
import ../ir/cirv2_lang
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerOptimizationPasses*(reg: var PassRegistry) =
  ## Registers the optimization stage passes, folding before dead-local
  ## elimination so the folds cascade into the deletions.
  reg.register("foldArithmeticIdentities", pkTransform, phaseMain,
    "Folds mul/div by the literal 1 and integer add/sub with 0 to the surviving operand",
    run = foldArithmeticIdentities)
  reg.register("foldSingleUseTemps", pkTransform, phaseMain,
    "Folds a once-assigned once-read temp into its only reader, deleting the store",
    run = foldSingleUseTemps)
  reg.register("removeDeadLocals", pkTransform, phaseMain,
    "Deletes the locals bound to constants nothing reads afterwards",
    run = eliminateDeadLocals)

# ═══════════════════════════════════════════════════════════════════════
# Arithmetic identity folding
# ═══════════════════════════════════════════════════════════════════════

func isLitValue(e: CirExpr, value: int64): bool =
  case e.kind
  of eIntLit: e.value == value
  of eFloatLit: e.fValue == float64(value)
  else: false

proc foldArithIdentity(e: CirExpr): CirExpr =
  ## Identity fold of one already-rewritten arithmetic node,
  ## nil when no identity applies.
  ##
  ## Soundness:
  ## - `x * 1` and `x / 1` are exact for every int and IEEE float value,
  ##   the bit pattern is preserved (including -0.0, NaN, Inf), so they fold
  ## - integer `add`/`sub` with 0 fold, float `add` with 0 does not:
  ##   `-0.0 + 0.0` is `+0.0` and the bit pattern changes
  ## - `x * 0` is not the identity (NaN, Inf), it never folds
  case e.kind
  of eMul:
    if isLitValue(e.left, 1):
      return e.right
    if isLitValue(e.right, 1):
      return e.left
  of eDiv:
    if isLitValue(e.right, 1):
      return e.left
  of eAdd, eSub:
    if e.opType.kind in {ctInt, ctUint}:
      if e.kind == eAdd:
        if isLitValue(e.left, 0):
          return e.right
        if isLitValue(e.right, 0):
          return e.left
      else:
        if isLitValue(e.right, 0):
          return e.left
  else:
    discard

proc foldIdentitiesExpr(e: CirExpr) =
  ## Folds the arithmetic identities in one expression, bottom-up,
  ## in place: a folded node's surviving operand moves into the node.
  if e.isNil:
    return
  case e.kind
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    foldIdentitiesExpr(e.left)
    foldIdentitiesExpr(e.right)
    let folded = foldArithIdentity(e)
    if folded != nil:
      e[] = folded[]
  of eAt:
    foldIdentitiesExpr(e.atBase)
    foldIdentitiesExpr(e.atIndex)
  of ePat:
    foldIdentitiesExpr(e.patBase)
    foldIdentitiesExpr(e.patIndex)
  of eDot:
    foldIdentitiesExpr(e.dotBase)
  of eDeref:
    foldIdentitiesExpr(e.derefOperand)
  of eAddr:
    foldIdentitiesExpr(e.addrOperand)
  of eEq, eNeq, eLt, eLe:
    foldIdentitiesExpr(e.a)
    foldIdentitiesExpr(e.b)
  of eAnd, eOr:
    foldIdentitiesExpr(e.andLeft)
    foldIdentitiesExpr(e.andRight)
  of eNot:
    foldIdentitiesExpr(e.notOperand)
  of eNeg, eBitNot:
    foldIdentitiesExpr(e.unOperand)
  of eCast, eConv:
    foldIdentitiesExpr(e.castOperand)
  of eOconstr:
    for f in e.kvFields.mitems:
      foldIdentitiesExpr(f.value)
  of eAconstr:
    for el in e.elems.mitems:
      foldIdentitiesExpr(el)
  of eCall:
    foldIdentitiesExpr(e.callee)
    for a in e.args.mitems:
      foldIdentitiesExpr(a)
  of eNil, eTrue, eFalse, eIntLit, eFloatLit, eSym, eSizeof:
    discard

proc foldIdentitiesStmts(ss: var seq[CirStmt]) =
  ## Folds the arithmetic identities through one statement list,
  ## nested bodies included.
  for s in ss:
    case s.kind
    of sVar, sConst:
      foldIdentitiesExpr(s.init)
    of sAsgn:
      foldIdentitiesExpr(s.lhs)
      foldIdentitiesExpr(s.rhs)
    of sIte:
      foldIdentitiesExpr(s.cond)
      foldIdentitiesStmts(s.thenBody)
      foldIdentitiesStmts(s.elseBody)
    of sLoop:
      foldIdentitiesStmts(s.loopBody)
    of sCase:
      foldIdentitiesExpr(s.selector)
      for b in s.branches.mitems:
        foldIdentitiesStmts(b.body)
      foldIdentitiesStmts(s.caseElse)
    of sScope:
      foldIdentitiesStmts(s.scopeBody)
    of sAssume:
      foldIdentitiesExpr(s.assumeCond)
    of sRet:
      foldIdentitiesExpr(s.retVal)
    of sDiscard:
      foldIdentitiesExpr(s.discardVal)
    of sCall:
      foldIdentitiesExpr(s.callCallee)
      for a in s.callArgs.mitems:
        foldIdentitiesExpr(a)
    of sAsm:
      for op in s.asmOperands.mitems:
        foldIdentitiesExpr(op)
    of sEmit:
      for op in s.emitOperands.mitems:
        foldIdentitiesExpr(op)
    of sLab, sJmp:
      discard

proc foldArithmeticIdentities(m: CirModule): string =
  for fn in m.procs:
    foldIdentitiesStmts(fn.body)
  foldIdentitiesStmts(m.globals)
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Single-use temp folding
# ═══════════════════════════════════════════════════════════════════════
#
# A temp folds into its only reader when the fold is sound.
#
# Candidate shape:
#
# - a per-thread register local, never `{.smem.}` or `{.const_mem.}`
# - one assignment, at declaration or by one `asgn` right after a bare declaration
# - one read, never address-taken, never an inline-assembly or `{.emit.}` operand
#   (those splice the name verbatim)
# - store and reader adjacent in one loop-free statement list, so no statement
#   reassigns an initializer operand in between and the read never executes
#   more often than the store
#
# Initializer movement:
#
# - a call-free initializer splices into an always-evaluated reader position:
#   an assignment right-hand side, a return value, a call argument, a condition, a selector, or a declaration initializer
# - the splice refuses when another call sits in that position. The emitted C
#   source evaluates call arguments in an unspecified order, so a sibling
#   call could write storage that the spliced initializer reads
# - a call initializer folds only into a whole destination read. The verifier admits
#   a call node nowhere else, so the callee must be side-effect free
# - calls with side effects (writes, barriers, atomics, subgroup reductions)
#   never move across anything, a doubtful case keeps the temp

proc exprHasSideEffects(e: CirExpr, procsByName: Table[string, CirProc], visiting: var HashSet[string]): bool
proc procHasSideEffects(fn: CirProc, procsByName: Table[string, CirProc], visiting: var HashSet[string]): bool

type
  SymUseCounts = object
    ## Per-symbol use counts over one proc body. Soundness input of the fold.
    reads: Table[string, int]
      ## Value reads per symbol, assignment targets excluded
    writes: Table[string, int]
      ## Plain `asgn`-target writes per symbol
    addrUses: HashSet[string]
      ## Symbols whose storage an `addr` names
    spliceUses: HashSet[string]
      ## Symbols an inline-assembly or `{.emit.}` statement splices by name

proc countExprUses(e: CirExpr, c: var SymUseCounts) =
  ## Counts one expression tree's symbol uses. Every symbol counts as a read,
  ## an `addr` operand and an assembly splice additionally mark their symbol.
  if e.isNil:
    return
  if e.kind == eSym:
    c.reads.mgetOrPut(e.name, 0).inc
    return
  if e.kind == eAddr and e.addrOperand != nil and e.addrOperand.kind == eSym:
    c.addrUses.incl e.addrOperand.name
  for ch in getExprChildren(e):
    countExprUses(ch, c)

proc countStmtUses(ss: seq[CirStmt], c: var SymUseCounts) =
  ## Counts one statement list's symbol uses. A read inside a loop or inside
  ## a nested body is still a read that the fold must see.
  for s in ss:
    case s.kind
    of sVar, sConst:
      countExprUses(s.init, c)
    of sAsgn:
      if s.lhs.kind == eSym:
        c.writes.mgetOrPut(s.lhs.name, 0).inc
      else:
        countExprUses(s.lhs, c)
      countExprUses(s.rhs, c)
    of sIte:
      countExprUses(s.cond, c)
      countStmtUses(s.thenBody, c)
      countStmtUses(s.elseBody, c)
    of sLoop:
      countStmtUses(s.loopBody, c)
    of sCase:
      countExprUses(s.selector, c)
      for b in s.branches:
        countStmtUses(b.body, c)
      countStmtUses(s.caseElse, c)
    of sScope:
      countStmtUses(s.scopeBody, c)
    of sAssume:
      countExprUses(s.assumeCond, c)
    of sRet:
      countExprUses(s.retVal, c)
    of sDiscard:
      countExprUses(s.discardVal, c)
    of sCall:
      countExprUses(s.callCallee, c)
      for a in s.callArgs:
        countExprUses(a, c)
    of sAsm:
      for op in s.asmOperands:
        if op.kind == eSym:
          c.spliceUses.incl op.name
        else:
          countExprUses(op, c)
    of sEmit:
      for op in s.emitOperands:
        if op.kind == eSym:
          c.spliceUses.incl op.name
        else:
          countExprUses(op, c)
    of sLab, sJmp:
      discard

proc containsCall(e: CirExpr): bool =
  ## True when the expression tree contains a call.
  if e.isNil:
    return false
  if e.kind == eCall:
    return true
  for ch in getExprChildren(e):
    if containsCall(ch):
      return true
  return false

proc containsSymbolRead(e: CirExpr): bool =
  ## True when the expression tree reads at least one symbol.
  if e.isNil:
    return false
  if e.kind == eSym:
    return true
  for ch in getExprChildren(e):
    if containsSymbolRead(ch):
      return true
  return false

proc findSymbolNode(e: CirExpr, name: string): CirExpr =
  ## Returns the symbol node named `name` in the expression tree,
  ## nil when the tree does not read it.
  if e.isNil:
    return nil
  if e.kind == eSym:
    if e.name == name:
      return e
    return nil
  for ch in getExprChildren(e):
    let found = findSymbolNode(ch, name)
    if found != nil:
      return found
  return nil

proc exprHasSideEffects(e: CirExpr, procsByName: Table[string, CirProc], visiting: var HashSet[string]): bool =
  ## True when evaluating the expression can write memory, synchronize,
  ## run an atomic or a subgroup reduction, or call a procedure of unknown effect.
  ## A coordinate builtin read evaluates no storage and stays pure.
  if e.isNil:
    return false
  if e.kind == eSym:
    return e.synchroBuiltin != sbkNone or e.atomicsBuiltin != abkNone or
      e.reductionBuiltin != rbkNone
  if e.kind == eCall:
    if e.callee.kind != eSym:
      return true
    if procsByName.hasKey(e.callee.name):
      if procHasSideEffects(procsByName[e.callee.name], procsByName, visiting):
        return true
    elif e.callee.builtin:
      if e.callee.synchroBuiltin != sbkNone or e.callee.atomicsBuiltin != abkNone or
          e.callee.reductionBuiltin != rbkNone:
        return true
    else:
      return true
    for a in e.args:
      if exprHasSideEffects(a, procsByName, visiting):
        return true
    return false
  for ch in getExprChildren(e):
    if exprHasSideEffects(ch, procsByName, visiting):
      return true
  return false

proc stmtsHaveSideEffects(ss: seq[CirStmt], procsByName: Table[string, CirProc], visiting: var HashSet[string]): bool =
  ## True when running the statement list can write memory outside the enclosing
  ## procedure's own locals. Synchronization, atomics, subgroup reductions,
  ## and calls to procedures of unknown effect all count.
  for s in ss:
    case s.kind
    of sAsm, sEmit:
      return true
    of sAsgn:
      if s.lhs.kind != eSym:
        return true
      if exprHasSideEffects(s.rhs, procsByName, visiting):
        return true
    of sVar, sConst:
      if exprHasSideEffects(s.init, procsByName, visiting):
        return true
    of sIte:
      if exprHasSideEffects(s.cond, procsByName, visiting):
        return true
      if stmtsHaveSideEffects(s.thenBody, procsByName, visiting) or
          stmtsHaveSideEffects(s.elseBody, procsByName, visiting):
        return true
    of sLoop:
      if stmtsHaveSideEffects(s.loopBody, procsByName, visiting):
        return true
    of sCase:
      if exprHasSideEffects(s.selector, procsByName, visiting):
        return true
      for b in s.branches:
        if stmtsHaveSideEffects(b.body, procsByName, visiting):
          return true
      if stmtsHaveSideEffects(s.caseElse, procsByName, visiting):
        return true
    of sScope:
      if stmtsHaveSideEffects(s.scopeBody, procsByName, visiting):
        return true
    of sAssume:
      if exprHasSideEffects(s.assumeCond, procsByName, visiting):
        return true
    of sRet:
      if exprHasSideEffects(s.retVal, procsByName, visiting):
        return true
    of sDiscard:
      if exprHasSideEffects(s.discardVal, procsByName, visiting):
        return true
    of sCall:
      if exprHasSideEffects(s.callCallee, procsByName, visiting):
        return true
      for a in s.callArgs:
        if exprHasSideEffects(a, procsByName, visiting):
          return true
    of sLab, sJmp:
      discard
  return false

proc procHasSideEffects(fn: CirProc, procsByName: Table[string, CirProc], visiting: var HashSet[string]): bool =
  ## True when running the procedure can write memory outside its own locals.
  ## Synchronization, atomics, subgroup reductions, and calls to procedures
  ## of unknown effect all count. A plain-symbol assignment rebinds a local
  ## or a value parameter, nothing outside observes it.
  if fn.name in visiting:
    return true
  visiting.incl fn.name
  return stmtsHaveSideEffects(fn.body, procsByName, visiting)

proc slotRoots(s: CirStmt): seq[CirExpr] =
  ## Returns the always-evaluated expression roots of one statement:
  ## the positions where a spliced initializer evaluates unconditionally,
  ## in evaluation order. A nested statement body is not a slot: a read there
  ## may run after the statement's own writes.
  case s.kind
  of sAsgn:
    result = @[s.rhs]
  of sRet:
    if s.retVal != nil:
      result = @[s.retVal]
  of sDiscard:
    if s.discardVal != nil:
      result = @[s.discardVal]
  of sAssume:
    result = @[s.assumeCond]
  of sCase:
    result = @[s.selector]
  of sIte:
    result = @[s.cond]
  of sVar, sConst:
    if s.init != nil:
      result = @[s.init]
  of sCall:
    result = s.callArgs
  of sLoop, sLab, sJmp, sAsm, sEmit, sScope:
    discard

proc tryFoldAt(ss: var seq[CirStmt], i: int, procsByName: Table[string, CirProc], c: SymUseCounts): bool =
  ## Folds the temp candidate stored at `ss[i]` into the single reader found
  ## at `ss[i + storeLen]`, deleting the store. True when a fold happened.
  ##
  ## The reader must sit immediately after the store, so no statement
  ## reassigns an initializer operand between them.
  var tempName = ""
  var init: CirExpr = nil
  var storeLen = 0
  let store = ss[i]
  if store.kind == sVar and store.space == caThread and store.name != "result":
    if store.init != nil:
      tempName = store.name
      init = store.init
      storeLen = 1
    elif i + 1 < ss.len and ss[i + 1].kind == sAsgn and
        ss[i + 1].lhs.kind == eSym and ss[i + 1].lhs.name == store.name:
      tempName = store.name
      init = ss[i + 1].rhs
      storeLen = 2
  if tempName.len == 0:
    return false
  let expectedWrites = if storeLen == 1: 0 else: 1
  if c.writes.getOrDefault(tempName, 0) != expectedWrites:
    return false
  if c.reads.getOrDefault(tempName, 0) != 1:
    return false
  if tempName in c.addrUses or tempName in c.spliceUses:
    return false
  if i + storeLen >= ss.len:
    return false
  let reader = ss[i + storeLen]
  if reader.kind == sLoop:
    return false

  if init.kind == eCall:
    # a call folds only into a whole destination read: the verifier admits
    # a call node nowhere else, and the callee must be side-effect free
    if not (reader.kind == sAsgn and reader.rhs.kind == eSym and
        reader.rhs.name == tempName):
      return false
    var visiting = initHashSet[string]()
    if exprHasSideEffects(init, procsByName, visiting):
      return false
    reader.rhs = init
  else:
    # a call-free initializer splices into an always-evaluated read slot,
    # never next to another call: the emitted C source evaluates call
    # arguments in an unspecified order
    let roots = reader.slotRoots()
    if roots.len == 0:
      return false
    if containsSymbolRead(init):
      for r in roots:
        if containsCall(r):
          return false
    var readNode: CirExpr = nil
    for r in roots:
      readNode = findSymbolNode(r, tempName)
      if readNode != nil:
        break
    if readNode.isNil:
      return false
    readNode[] = init[]

  ss.delete(i, i + storeLen - 1)
  return true

proc foldSingleUseTempsInList(ss: var seq[CirStmt], procsByName: Table[string, CirProc], c: SymUseCounts): bool =
  ## Folds at most one temp in this statement list. Nested loop-free lists
  ## are also scanned: a loop body is never entered, so a temp read inside
  ## a loop keeps its store. True when a fold happened. The caller iterates
  ## to a fixpoint over fresh use counts.
  for i in 0 ..< ss.len:
    if tryFoldAt(ss, i, procsByName, c):
      return true
    case ss[i].kind
    of sIte:
      if foldSingleUseTempsInList(ss[i].thenBody, procsByName, c):
        return true
      if foldSingleUseTempsInList(ss[i].elseBody, procsByName, c):
        return true
    of sCase:
      for bi in 0 ..< ss[i].branches.len:
        if foldSingleUseTempsInList(ss[i].branches[bi].body, procsByName, c):
          return true
      if foldSingleUseTempsInList(ss[i].caseElse, procsByName, c):
        return true
    of sScope:
      if foldSingleUseTempsInList(ss[i].scopeBody, procsByName, c):
        return true
    else:
      discard
  return false

proc foldSingleUseTemps(m: CirModule): string =
  ## Folds every once-assigned once-read temp into its only reader. The pass iterates
  ## to a fixpoint.
  ##
  ## Soundness contract:
  ##
  ## - the store and the reader are adjacent statements of one loop-free
  ##   statement list
  ## - the initializer is call-free or a side-effect-free call
  ## - a call initializer folds only into a whole destination read
  ##
  ## Returns "" on every input, this pass never refuses.
  var procsByName = initTable[string, CirProc]()
  for fn in m.procs:
    procsByName[fn.name] = fn
  const maxRounds = 64
  for round in 0 ..< maxRounds:
    var folded = false
    for fn in m.procs:
      var c = SymUseCounts()
      countStmtUses(fn.body, c)
      if foldSingleUseTempsInList(fn.body, procsByName, c):
        folded = true
    if not folded:
      break
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Dead-local elimination
# ═══════════════════════════════════════════════════════════════════════

proc eliminateDeadLocals(m: CirModule): string =
  ## Deletes every local bound to a constant that nothing reads, to a fixpoint.
  while m.procs.anyIt(removeDeadLocals(it)):
    discard
  return ""


proc collectLocalReads(e: CirExpr, reads: var HashSet[string]) =
  if e.isNil:
    return
  if e.kind == eSym:
    reads.incl e.name
    return
  for ch in getExprChildren(e):
    collectLocalReads(ch, reads)

proc collectLocalReads(stmts: seq[CirStmt], reads: var HashSet[string]) =
  for s in stmts:
    case s.kind
    of sVar, sConst:
      collectLocalReads(s.init, reads)
    of sAsgn:
      collectLocalReads(s.lhs, reads)
      collectLocalReads(s.rhs, reads)
    of sIte:
      collectLocalReads(s.cond, reads)
      collectLocalReads(s.thenBody, reads)
      collectLocalReads(s.elseBody, reads)
    of sLoop:
      collectLocalReads(s.loopBody, reads)
    of sScope:
      collectLocalReads(s.scopeBody, reads)
    of sAssume:
      collectLocalReads(s.assumeCond, reads)
    of sDiscard:
      collectLocalReads(s.discardVal, reads)
    of sRet:
      collectLocalReads(s.retVal, reads)
    of sCall:
      collectLocalReads(s.callCallee, reads)
      for a in s.callArgs:
        collectLocalReads(a, reads)
    of sAsm:
      # an asm operand reads its symbol, so the dead-local pass keeps it
      for op in s.asmOperands:
        collectLocalReads(op, reads)
    of sEmit:
      # an emit operand reads its symbol, so the dead-local pass keeps it
      for op in s.emitOperands:
        collectLocalReads(op, reads)
    else:
      discard

proc removeDeadLocals(fn: CirProc): bool =
  ## Deletes the locals that nothing reads, their writes included, one round.
  ## True when something was removed. The caller iterates to a fixpoint.
  var reads = initHashSet[string]()
  fn.body.collectLocalReads(reads)
  var removed = false
  var kept: seq[CirStmt]
  for s in fn.body:
    case s.kind
    of sVar, sConst:
      if s.name notin reads and s.name != "result":
        removed = true
        continue
      kept.add s
    of sAsgn:
      if s.lhs.kind == eSym and s.lhs.name notin reads and s.lhs.name != "result":
        removed = true
        continue
      kept.add s
    else:
      kept.add s
  fn.body = kept
  return removed
