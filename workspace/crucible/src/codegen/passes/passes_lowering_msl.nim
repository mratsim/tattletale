# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## MSL lowering: per-target transforms for the Metal storage model.
## The pipeline table in `gpu_compiler.nim` declares the registration order:
##
##   legalize → normalize → lowerSpans → mslReservedIdentifiers → decomposeArrayAssign
##   → explodeGlobalStructWithPtrFields → smemInitRewrite → mslHoistProgramScopeConstants
##   → mslHoistDeviceFnSmem → mslRewriteSmemLocals → mslThreadCoordinateBuiltins
##   → instantiateAddressSpaceGenerics → validation
##
## - `mslHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into `CirModule.globals`
##   in encounter order for the printer to emit as program-scope `constant`
##   variables visible to all functions
## - `mslHoistDeviceFnSmem`: a device function's `{.smem.}` local hoists
##   to a threadgroup reference parameter, Metal supports no threadgroup
##   allocation in device functions, a hoisted arena takes no initializer
## - `mslRewriteSmemLocals`: the pass keeps a kernel's `{.smem.}` local at any
##   body depth in-body as kernel-scope `threadgroup` storage.
##   the pass refuses an initializer: Metal drops threadgroup array initializers at runtime
## - `mslThreadCoordinateBuiltins`: coordinate builtins thread down to device
##   functions as parameters (Metal rejects them outside kernel functions)
##
## Refusal homes:
## - the pass refuses a const_mem local without an initializer, the printer
##   refuses a survivor outside the relocation, keeping a nil-init shape
##   guard on the relocated declarations
## - the pass refuses a device function's smem local outside the hoist's
##   reach in `mslRewriteSmemLocals`, only a kernel owns threadgroup storage
## - `mslHoistDeviceFnSmem` refuses a smem initializer on a hoisted arena,
##   the arena is the caller's storage to lend, a kernel's own smem
##   initializer rewrote to stores before the pass
## - `mslThreadCoordinateBuiltins` refuses a device function
##   declaring a parameter named like a needed coordinate builtin, as the call sites could
##   not supply both values under one name

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerMslHoistProgramScopeConstants*(reg: var PassRegistry) =
  reg.register("mslHoistProgramScopeConstants", pkTransform,
    "Relocates const_mem locals into the module's program scope",
    run = mslHoistProgramScopeConstants
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
  ## Each pass of the MSL window appends its parameters and matching
  ## call-site arguments last, in registration order.
  reg.register("mslThreadCoordinateBuiltins", pkTransform,
    "Threads coordinate builtins down to device functions as parameters," &
    " call sites forward the caller's values",
    run = mslThreadCoordinateBuiltins
  )

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program-scope constants
# ═══════════════════════════════════════════════════════════════════════

proc mslHoistProgramScopeConstants(m: CirModule): string =
  hoistConstMemLocals(m)

# ═══════════════════════════════════════════════════════════════════════
# Device-function `{.smem.}` locals to threadgroup reference parameters
# ═══════════════════════════════════════════════════════════════════════

proc extendedProcType(t: CirType, extra: seq[CirType]): CirType =
  ## Callee types are interned, structurally identical signatures share one
  ## `CirType` ref: an in-place extension would leak into every proc and call
  ## site carrying the signature, so each call site takes a fresh extended copy.
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

proc hoistSmemLocal(fn: CirProc, s: CirStmt, arenas: var seq[SmemArena]): string =
  ## An arena is the caller's storage to lend, the pass refuses an initializer on it.
  if s.init != nil:
    return "at " & fn.name & ": smem local " & s.name &
      " with an initializer, a hoisted {.smem.} arena takes no initializer"
  let typ = if s.declType.kind == ctArray:
    tAptr(caThreadgroup, s.declType.aElem)
  else:
    tPtr(caThreadgroup, s.declType)
  fn.params.add(param(s.name, typ))
  arenas.add((s.name, s.declType, typ))

proc hoistSmemLocals(fn: CirProc, arenas: var seq[SmemArena]): string =
  var i = 0
  while i < fn.body.len:
    let s = fn.body[i]
    if s.kind == sVar and s.space == caThreadgroup:
      let err = fn.hoistSmemLocal(s, arenas)
      if err.len > 0:
        return err
      fn.body.delete i
      continue
    i.inc
  return ""

proc rewriteArenaReads(e: CirExpr, arenaTypes: Table[string, CirType]) =
  ## An `at` over a hoisted array arena becomes a `pat`, the rewrite prints `scratch[i]`.
  ## Expressions are refs, so the rewrite applies to every sharer.
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
    if p.typ.kind == ctPtr and p.typ.space == caThreadgroup:
      result.incl(p.name)
  for s in fn.body:
    if s.kind == sVar and s.space == caThreadgroup:
      result.incl(s.name)

proc bindSmemArgs(fn: CirProc, calleeProc: CirProc, callee: CirExpr,
                  args: var seq[CirExpr],
                  arenas: seq[SmemArena], owned: var HashSet[string]): string =
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
    return "at " & fn.name & ": device fn '" & callee.name &
      "' owns the {.smem.} arena '" & arena.name &
      "', the caller must declare a matching {.smem.} var"
  return ""

proc bindSmemCallSiteStmts(fn: CirProc, ss: seq[CirStmt], procs: Table[string, CirProc], hoisted: Table[string, seq[SmemArena]]): string =
  ## Binding a kernel call site synthesizes its arena declaration into `fn.body`.
  ## The synthesized declaration binds no call, so the walk need not visit it.
  var owned = fn.ownedSmemNames()
  var err = ""
  let bindCall = proc(callee: CirExpr, args: var seq[CirExpr]) =
    if callee.kind == eSym and hoisted.hasKey(callee.name):
      err = bindSmemArgs(fn, procs[callee.name], callee, args,
        hoisted[callee.name], owned)
  let onStmt = proc(s: CirStmt) =
    if err.len > 0:
      return
    if s.kind == sCall:
      s.callCallee.bindCall(s.callArgs)
    elif s.kind == sAsgn and s.rhs.kind == eCall:
      s.rhs.callee.bindCall(s.rhs.args)
  ss.walkStmts(onStmt, nil)
  return err

proc mslHoistDeviceFnSmem(m: CirModule): string =
  ## Runs before `mslRewriteSmemLocals`, which keeps the kernel-owned arenas in-body.
  var hoisted = initTable[string, seq[SmemArena]]()
  for fn in m.procs:
    if fn.isKernel:
      continue
    var arenas: seq[SmemArena]
    let hoistErr = fn.hoistSmemLocals(arenas)
    if hoistErr.len > 0:
      return hoistErr
    if arenas.len == 0:
      continue
    hoisted[fn.name] = arenas
    var arenaTypes = initTable[string, CirType]()
    for arena in arenas:
      if arena.declType.kind == ctArray:
        arenaTypes[arena.name] = arena.typ
    fn.body.rewriteArenaReadsStmts(arenaTypes)
  if hoisted.len == 0:
    return ""
  let procs = procsByName(m)
  for fn in m.procs:
    let err = bindSmemCallSiteStmts(fn, fn.body, procs, hoisted)
    if err.len > 0:
      return err
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins threaded down to device functions
# ═══════════════════════════════════════════════════════════════════════

proc collectCoordReads(e: CirExpr, kinds: var HashSet[CoordBuiltinKind]) =
  if e.kind == eSym and e.coordBuiltin != cbkNone:
    kinds.incl(e.coordBuiltin)

proc collectCoordReadsStmts(ss: seq[CirStmt]): HashSet[CoordBuiltinKind] =
  var kinds: HashSet[CoordBuiltinKind]
  let collect = proc(e: CirExpr) =
    e.collectCoordReads(kinds)
  ss.walkStmts(nil, collect)
  return kinds

proc getBuiltinParamType(kind: CoordBuiltinKind): CirType =
  ## Builtins carry the canonical `ivec3` type, so the printer emits an `int3` constructor for whole-value reads of the standing-in parameter.
  if kind == cbkThreadIndexInThreadgroup:
    tInt(32)
  else:
    tVec(3, tInt(32))

proc directCoordNeeds(fn: CirProc, direct: Table[string, HashSet[CoordBuiltinKind]],
                      callTargets: Table[string, HashSet[string]],
                      needs: var Table[string, HashSet[CoordBuiltinKind]]): bool =
  var kinds = direct[fn.name]
  for target in callTargets[fn.name]:
    if needs.hasKey(target):
      for kind in needs[target]:
        kinds.incl(kind)
  if kinds == needs.getOrDefault(fn.name):
    return false
  needs[fn.name] = kinds
  return true

proc sortCoordKinds(kinds: HashSet[CoordBuiltinKind]): seq[CoordBuiltinKind] =
  ## Needed kinds in declaration order, which also fixes the parameter order.
  for kind in CoordBuiltinKind:
    if kind != cbkNone and kind in kinds:
      result.add(kind)

proc appendCoordParam(fn: CirProc, kind: CoordBuiltinKind): string =
  ## A parameter takes the builtin's canonical name so the body's
  ## builtin-marked reads print that name unchanged, in a kernel the name
  ## binds to the attributed parameter, in a device function to this parameter.
  let name = CoordBuiltinCanonicalName[kind]
  for p in fn.params:
    if p.name == name:
      return "at " & fn.name & ": parameter " & name &
        " is named like the coordinate builtin the function needs," &
        " the call sites cannot supply both"
  fn.params.add(param(name, getBuiltinParamType(kind)))
  return ""

proc appendCoordCallArgs(fn: CirProc, calleeProc: CirProc, callee: CirExpr,
                         args: var seq[CirExpr],
                         needs: Table[string, HashSet[CoordBuiltinKind]]): string =
  ## One argument per needed builtin in the callee's parameter order, the callee symbol's proc type grows to match.
  ##
  ## - a kernel forwards the builtin-marked symbol: the printer then emits
  ##   the builtin read and the kernel signature gains the attributed parameter
  ## - a device function forwards its own parameter for the printer to emit
  ##   as the bare name
  if callee.kind != eSym or not needs.hasKey(callee.name):
    return ""
  let calleeKinds = sortCoordKinds(needs[callee.name])
  if callee.typ.kind == ctProc and
      callee.typ.params.len == calleeProc.params.len - calleeKinds.len:
    # one extension per callee symbol, call sites can share the node
    var extra: seq[CirType]
    for kind in calleeKinds:
      extra.add(getBuiltinParamType(kind))
    callee.typ = extendedProcType(callee.typ, extra)
  for kind in calleeKinds:
    if fn.isKernel:
      args.add(sym(CoordBuiltinCanonicalName[kind], getBuiltinParamType(kind),
        builtin = true, coordBuiltin = kind))
    else:
      args.add(sym(CoordBuiltinCanonicalName[kind], getBuiltinParamType(kind)))
  return ""

proc appendCoordForwardingArgsStmts(fn: CirProc, ss: seq[CirStmt],
                                    procs: Table[string, CirProc],
                                    needs: Table[string, HashSet[CoordBuiltinKind]]): string =
  var err = ""
  let onExpr = proc(e: CirExpr) =
    if err.len > 0:
      return
    if e.kind == eCall:
      err = appendCoordCallArgs(fn, procs.getOrDefault(e.callee.name), e.callee,
        e.args, needs)
  let onStmt = proc(s: CirStmt) =
    if err.len > 0:
      return
    if s.kind == sCall:
      err = appendCoordCallArgs(fn, procs.getOrDefault(s.callCallee.name),
        s.callCallee, s.callArgs, needs)
  ss.walkStmts(onStmt, onExpr)
  return err

proc mslThreadCoordinateBuiltins(m: CirModule): string =
  ## - every device function gains one parameter per transitively needed coordinate builtin in builtin-kind declaration order
  ## - a kernel body is untouched, it reads the builtins it forwards directly
  ## - Metal only: the builtins are legal reads inside device functions
  ##   on other targets, so the pass is not registered there
  var needs = initTable[string, HashSet[CoordBuiltinKind]]()
  var direct = initTable[string, HashSet[CoordBuiltinKind]]()
  var callTargets = initTable[string, HashSet[string]]()
  for fn in m.procs:
    needs[fn.name] = initHashSet[CoordBuiltinKind]()
    direct[fn.name] = fn.body.collectCoordReadsStmts()
    var targets: HashSet[string]
    fn.body.collectCallTargets(targets)
    callTargets[fn.name] = targets
  # fixpoint over the call graph: a builtin needed deep in a callee chain
  # reaches every caller, and the round count bounds cyclic call graphs
  var changed = true
  var rounds = 0
  while changed:
    changed = false
    for fn in m.procs:
      if fn.directCoordNeeds(direct, callTargets, needs):
        changed = true
    rounds.inc
    if rounds > m.procs.len + 1:
      return "at " & m.name & ": coordinate-builtin needs do not converge"
  var anyNeeds = false
  for fn in m.procs:
    if needs[fn.name].len > 0:
      anyNeeds = true
      break
  if not anyNeeds:
    return ""
  for fn in m.procs:
    if fn.isKernel:
      continue
    for kind in sortCoordKinds(needs[fn.name]):
      let err = fn.appendCoordParam(kind)
      if err.len > 0:
        return err
  let procs = procsByName(m)
  for fn in m.procs:
    let err = appendCoordForwardingArgsStmts(fn, fn.body, procs, needs)
    if err.len > 0:
      return err
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Kernel smem locals to kernel-scope threadgroup storage
# ═══════════════════════════════════════════════════════════════════════

proc mslRewriteSmemLocals(m: CirModule): string =
  ## A kernel keeps its `{.smem.}` local in its body, top-level or nested:
  ##
  ## - the printer emits kernel-scope `threadgroup` storage, allocated per
  ##   threadgroup by Metal, no launch-time staging and no parameter
  ## - the pass refuses a device function's smem local, only a kernel owns
  ##   threadgroup storage
  ##
  ## - the hoist drains the locals it covers, this walk catches a survivor at any depth
  ## - the initializer rewrite ran earlier, a surviving local has no initializer
  ## - the pass runs after the reserved-word rename, so the kept locals use the renamed names their body already references
  for fn in m.procs:
    var err = ""
    let kernel = fn.isKernel
    let procName = fn.name
    let onStmt = proc(s: CirStmt) =
      if s.kind != sVar or s.space != caThreadgroup:
        return
      if not kernel:
        err = "at " & procName & ": smem local " & s.name &
          " in a device function, only a kernel allocates" &
          " threadgroup storage"
    fn.body.walkStmts(onStmt, nil)
    if err.len > 0:
      return err
  return ""
