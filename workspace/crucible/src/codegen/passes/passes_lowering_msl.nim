# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ############################################################
#
#            MSL lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerMslHoistProgramScopeConstants*(reg: var PassRegistry) =
  reg.register("mslHoistProgramScopeConstants", pkTransform,
    "Relocates const_mem locals into the module's program scope",
    run = hoistConstMemLocals
  )

func registerMslHoistDeviceFnSmem*(reg: var PassRegistry) =
  reg.register("mslHoistDeviceFnSmem", pkTransform,
    "Hoists a device function's smem locals to threadgroup reference parameters," &
    " call sites pass the caller's arena",
    run = mslHoistDeviceFnSmem
  )

func registerMslRewriteSmemLocals*(reg: var PassRegistry) =
  reg.register("mslRewriteSmemLocals", pkTransform,
    "Keeps kernel smem locals in-body as threadgroup storage, refuses a device function's smem local",
    run = mslRewriteSmemLocals
  )

func registerMslThreadCoordinateBuiltins*(reg: var PassRegistry) =
  ## Each MSL-window pass appends its parameters and matching call-site arguments last, in registration order.
  reg.register("mslThreadCoordinateBuiltins", pkTransform,
    "Threads coordinate builtins down to device functions as parameters," &
    " call sites forward the caller's values",
    run = mslThreadCoordinateBuiltins
  )

# ═══════════════════════════════════════════════════════════════════════
#  Smem reference parameters
# ═══════════════════════════════════════════════════════════════════════

proc extendedProcType(t: CirType, extra: seq[CirType]): CirType =
  result = CirType(kind: ctProc)
  result.params = t.params & extra
  result.ret = t.ret

proc procsByName(m: CirModule): Table[string, CirProc] =
  for fn in m.procs:
    result[fn.name] = fn

type
  SmemArena = tuple[name: string, declType: CirType, typ: CirType]
    ## One hoisted `{.smem.}` local:
    ## - `name`, `declType`: the parameter name and the declared local type
    ## - `typ`: the threadgroup pointer type that replaced the local, an array
    ##   arena takes an `aptr` over the element, an object arena a `ptr` over
    ##   the object

proc hoistSmemLocal(arenas: var seq[SmemArena], fn: CirProc, s: CirStmt) =
  if s.init != nil:
    raiseAssert("smem local " & s.name &
      " with an initializer, a hoisted {.smem.} arena takes no initializer")
  let typ = if s.declType.kind == ctArray:
    tAptr(caThreadgroup, s.declType.aElem)
  else:
    tPtr(caThreadgroup, s.declType)
  fn.params.add(param(s.name, typ))
  arenas.add((s.name, s.declType, typ))

proc hoistSmemLocals(arenas: var seq[SmemArena], fn: CirProc) =
  var i = 0
  while i < fn.body.len:
    let s = fn.body[i]
    if s.kind == sVar and s.addressSpace == caThreadgroup:
      hoistSmemLocal(arenas, fn, s)
      fn.body.delete i
      continue
    i.inc

proc rewriteArenaReads(e: CirExpr, arenaTypes: Table[string, CirType]) =
  if e.kind == eSym and arenaTypes.hasKey(e.name):
    e.typ = arenaTypes[e.name]
  if e.kind == eAt and e.atBase.kind == eSym and arenaTypes.hasKey(e.atBase.name):
    let base = e.atBase
    let idx = e.atIndex
    base.typ = arenaTypes[base.name]
    e.kind = ePat
    e.patBase = base
    e.patIndex = idx

proc rewriteArenaReadsStmts(ss: seq[CirStmt], arenaTypes: Table[string, CirType]) =
  ss.walkStmts(nil, proc(e: CirExpr) =
    e.rewriteArenaReads(arenaTypes))

proc ownedSmemNames(fn: CirProc): HashSet[string] =
  for p in fn.params:
    if p.typ.kind == ctPtr and p.typ.addressSpace == caThreadgroup:
      result.incl(p.name)
  for s in fn.body:
    if s.kind == sVar and s.addressSpace == caThreadgroup:
      result.incl(s.name)

proc bindSmemArgs(args: var seq[CirExpr], owned: var HashSet[string],
                  fn: CirProc, calleeProc: CirProc, callee: CirExpr,
                  arenas: seq[SmemArena]) =
  ## - a matching-name `{.smem.}` local in the caller supplies the argument,
  ##   an array arena decays to its element pointer in the emitted source,
  ##   an object arena binds by reference
  ## - a kernel caller without one gets the arena synthesized into its body
  ##   as the first statement (the launchDeviceFn wrapper)
  ## - the pass refuses a device-function caller without one, threadgroup
  ##   storage is the caller's to lend
  ##
  ## A callee symbol's proc type grows by the arena types, idempotent per shared symbol node and matched to the appended arguments.
  if callee.typ.kind == ctProc and
      callee.typ.params.len == calleeProc.params.len - arenas.len:
    var extra: seq[CirType]
    for arena in arenas:
      extra.add(arena.typ)
    callee.typ = extendedProcType(callee.typ, extra)
  for arena in arenas:
    if arena.name in owned:
      args.add(sym(arena.name, arena.typ))
      continue
    if fn.isKernel:
      fn.body.insert(varD(arena.name, arena.declType, nil, caThreadgroup), 0)
      owned.incl(arena.name)
      args.add(sym(arena.name, arena.typ))
      continue
    raiseAssert("device fn '" & callee.name &
      "' owns the {.smem.} arena '" & arena.name &
      "', the caller must declare a matching {.smem.} var")

proc bindSmemCallSiteStmts(fn: CirProc, ss: seq[CirStmt], procs: Table[string, CirProc], hoisted: Table[string, seq[SmemArena]]) =
  ## Binding a kernel call site synthesizes its arena declaration into `fn.body`.
  ## The synthesized declaration binds no call, so the walk need not visit it.
  var owned = fn.ownedSmemNames()
  let bindCall = proc(callee: CirExpr, args: var seq[CirExpr]) =
    if callee.kind == eSym and hoisted.hasKey(callee.name):
      bindSmemArgs(args, owned, fn, procs[callee.name], callee,
        hoisted[callee.name])
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall:
      s.callCallee.bindCall(s.callArgs)
    elif s.kind == sAsgn and s.rhs.kind == eCall:
      s.rhs.callee.bindCall(s.rhs.args)
  ss.walkStmts(onStmt, nil)

proc mslHoistDeviceFnSmem(m: CirModule) =
  ## Runs before `mslRewriteSmemLocals`, which keeps the kernel-owned arenas in-body.
  var hoisted = initTable[string, seq[SmemArena]]()
  for fn in m.procs:
    if fn.isKernel:
      continue
    var arenas: seq[SmemArena]
    hoistSmemLocals(arenas, fn)
    if arenas.len == 0:
      continue
    hoisted[fn.name] = arenas
    var arenaTypes = initTable[string, CirType]()
    for arena in arenas:
      if arena.declType.kind == ctArray:
        arenaTypes[arena.name] = arena.typ
    fn.body.rewriteArenaReadsStmts(arenaTypes)
  if hoisted.len == 0:
    return
  let procs = procsByName(m)
  for fn in m.procs:
    bindSmemCallSiteStmts(fn, fn.body, procs, hoisted)

# ═══════════════════════════════════════════════════════════════════════
#  Device-function coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc collectCoordReads(kinds: var HashSet[CirCoordBuiltinKind], e: CirExpr) =
  if e.kind == eSym and e.builtin and e.coordBuiltin != cbkNone:
    kinds.incl(e.coordBuiltin)

proc collectCoordReadsStmts(ss: seq[CirStmt]): HashSet[CirCoordBuiltinKind] =
  var kinds: HashSet[CirCoordBuiltinKind]
  let collect = proc(e: CirExpr) =
    collectCoordReads(kinds, e)
  ss.walkStmts(nil, collect)
  return kinds

proc getBuiltinParamType(k: CirCoordBuiltinKind): CirType =
  if k == cbkThreadIndexInThreadgroup:
    tInt(32)
  else:
    tVec(3, tInt(32))

proc directCoordNeeds(needs: var Table[string, HashSet[CirCoordBuiltinKind]],
                      fn: CirProc, direct: Table[string, HashSet[CirCoordBuiltinKind]],
                      callTargets: Table[string, HashSet[string]]): bool =
  var kinds = direct[fn.name]
  for target in callTargets[fn.name]:
    if needs.hasKey(target):
      for k in needs[target]:
        kinds.incl(k)
  if kinds == needs.getOrDefault(fn.name):
    return false
  needs[fn.name] = kinds
  return true

proc orderedCoordKinds(kinds: HashSet[CirCoordBuiltinKind]): seq[CirCoordBuiltinKind] =
  for k in cbkThreadPositionInGrid..cbkThreadIndexInThreadgroup:
    if k in kinds:
      result.add(k)

proc appendCoordParam(fn: CirProc, k: CirCoordBuiltinKind) =
  ## A parameter uses the builtin's canonical identifier so the body's
  ## builtin-marked reads print it unchanged, in a kernel the identifier
  ## binds to the attributed parameter, in a device function to this parameter.
  let name = $k
  for p in fn.params:
    if p.name == name:
      raiseAssert("parameter " & name &
        " is named like the coordinate builtin the function needs," &
        " the call sites cannot supply both")
  fn.params.add(param(name, getBuiltinParamType(k)))

proc appendCoordCallArgs(args: var seq[CirExpr], fn: CirProc, calleeProc: CirProc, callee: CirExpr,
                         needs: Table[string, HashSet[CirCoordBuiltinKind]]) =
  ## One argument per needed builtin in the callee's parameter order, the callee symbol's proc type grows to match.
  ##
  ## - a kernel forwards the builtin-marked symbol with its kind included
  ## - a device function forwards its own parameter, printed as its own name
  if callee.kind != eSym or not needs.hasKey(callee.name):
    return
  let calleeKinds = orderedCoordKinds(needs[callee.name])
  if callee.typ.kind == ctProc and
      callee.typ.params.len == calleeProc.params.len - calleeKinds.len:
    # one extension per callee symbol, call sites can share the node
    var extra: seq[CirType]
    for k in calleeKinds:
      extra.add(getBuiltinParamType(k))
    callee.typ = extendedProcType(callee.typ, extra)
  for k in calleeKinds:
    let arg = sym($k, getBuiltinParamType(k), builtin = true)
    arg.coordBuiltin = k
    args.add(arg)

proc appendCoordForwardingArgsStmts(fn: CirProc, ss: seq[CirStmt],
                                    procs: Table[string, CirProc],
                                    needs: Table[string, HashSet[CirCoordBuiltinKind]]) =
  let onExpr = proc(e: CirExpr) =
    if e.kind == eCall:
      appendCoordCallArgs(e.args, fn, procs.getOrDefault(e.callee.name), e.callee,
        needs)
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall:
      appendCoordCallArgs(s.callArgs, fn, procs.getOrDefault(s.callCallee.name),
        s.callCallee, needs)
  ss.walkStmts(onStmt, onExpr)

proc mslThreadCoordinateBuiltins(m: CirModule) =
  ## - every device function gains one parameter per transitively needed coordinate builtin in builtin-kind declaration order
  ## - a kernel body is untouched, it reads the builtins it forwards directly
  ## - Metal only: the builtins are legal reads inside device functions
  ##   on other targets, so the pass is not registered there
  var needs = initTable[string, HashSet[CirCoordBuiltinKind]]()
  var direct = initTable[string, HashSet[CirCoordBuiltinKind]]()
  var callTargets = initTable[string, HashSet[string]]()
  for fn in m.procs:
    needs[fn.name] = initHashSet[CirCoordBuiltinKind]()
    direct[fn.name] = fn.body.collectCoordReadsStmts()
    var targets: HashSet[string]
    collectCallTargets(targets, fn.body)
    callTargets[fn.name] = targets
  # fixpoint over the call graph: a builtin needed deep in a callee chain
  # reaches every caller, and the round count bounds cyclic call graphs
  var changed = true
  var rounds = 0
  while changed:
    changed = false
    for fn in m.procs:
      if directCoordNeeds(needs, fn, direct, callTargets):
        changed = true
    rounds.inc
    if rounds > m.procs.len + 1:
      raiseAssert("coordinate-builtin needs do not converge")
  var anyNeeds = false
  for fn in m.procs:
    if needs[fn.name].len > 0:
      anyNeeds = true
      break
  if not anyNeeds:
    return
  for fn in m.procs:
    if fn.isKernel:
      continue
    for k in orderedCoordKinds(needs[fn.name]):
      fn.appendCoordParam(k)
  let procs = procsByName(m)
  for fn in m.procs:
    appendCoordForwardingArgsStmts(fn, fn.body, procs, needs)

# ═══════════════════════════════════════════════════════════════════════
#  Kernel threadgroup storage
# ═══════════════════════════════════════════════════════════════════════

proc mslRewriteSmemLocals(m: CirModule) =
  ## A kernel keeps its `{.smem.}` local in its body, top-level or nested:
  ##
  ## - the printer emits kernel-scope `threadgroup` storage, allocated per
  ##   threadgroup by Metal, no launch-time staging and no parameter
  ## - the pass refuses a device function's smem local, only a kernel owns
  ##   threadgroup storage
  ##
  ## - the hoist drains the locals it covers, this walk refuses one remaining at any depth
  ## - the initializer rewrite ran earlier, a surviving local has no initializer
  ## - the pass runs after the reserved-word rename, so the kept locals use the renamed names their body already references
  for fn in m.procs:
    let kernel = fn.isKernel
    let onStmt = proc(s: CirStmt) =
      if s.kind != sVar or s.addressSpace != caThreadgroup:
        return
      if not kernel:
        raiseAssert("smem local " & s.name &
          " in a device function, only a kernel allocates" &
          " threadgroup storage")
    fn.body.walkStmts(onStmt, nil)
