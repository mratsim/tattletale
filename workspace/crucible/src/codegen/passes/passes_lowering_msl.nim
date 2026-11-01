# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## MSL lowering: per-target transforms for the Metal storage model,
## registered by the Metal pipeline and run before printing.
##
##   kernel body → decomposeArrayAssign → mslHoistProgramScopeConstants
##   → mslHoistDeviceFnSmem → mslRewriteSmemLocals → mslThreadCoordinateBuiltins
##   → printed MSL source
##
## - `mslHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into `CirModule.globals`
##   in encounter order, the printer emits them as program-scope `constant`
##   variables visible to all functions
## - `mslHoistDeviceFnSmem`: a device function's `{.smem.}` local hoists
##   to a threadgroup reference parameter, Metal supports no threadgroup
##   allocation in device functions
## - `mslRewriteSmemLocals`: a kernel's `{.smem.}` local at any body depth
##   stays in-body as kernel-scope `threadgroup` storage
## - `mslThreadCoordinateBuiltins`: coordinate builtins thread down to device
##   functions as parameters (Metal rejects them outside kernel functions)
##
## Refusal homes:
## - a const_mem local without an initializer refuses in the pass, a survivor
##   outside the relocation refuses at the printer, which keeps a nil-init
##   shape guard on the relocated declarations
## - a device function's smem local outside the hoist's reach refuses in `mslRewriteSmemLocals`,
##   only a kernel owns threadgroup storage
## - a smem initializer surviving the pass is discarded at the printer
## - a device function declaring a parameter named like a needed coordinate
##   builtin refuses in `mslThreadCoordinateBuiltins`, the call sites could
##   not supply both values under one name

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers
import ./passes_low_address_spaces
import ./passes_low_array_assign
import ./passes_low_pointers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerMslLoweringPasses*(reg: var PassRegistry) =
  ## Registers the MSL per-target lowering passes:
  ## - `decomposeArrayAssign`: whole-array copies that MSL cannot express
  ## - the kernel struct-argument explosion, the flat resource target
  ## - `mslHoistProgramScopeConstants`, `mslHoistDeviceFnSmem`,
  ##   `mslRewriteSmemLocals`, `mslThreadCoordinateBuiltins`, in that order,
  ##   before printing
  ##
  ## Registration contract: the passes sit after the reserved-word rename,
  ## relocated declaration and appended parameter names match the renamed body references.
  ##
  ## Each pass appends its parameters and matching call-site arguments last,
  ## the two orders stay consistent.
  reg.registerDecomposeArrayAssign()
  registerExplodeStructWithPtrFields(reg)
  reg.register("mslHoistProgramScopeConstants", pkTransform,
    "Relocates const_mem locals into the module's program scope",
    run = mslHoistProgramScopeConstants
  )
  reg.register("mslHoistDeviceFnSmem", pkTransform,
    "Hoists a device function's smem locals to threadgroup reference parameters," &
    " call sites pass the caller's arena",
    run = mslHoistDeviceFnSmem
  )
  reg.register("mslRewriteSmemLocals", pkTransform,
    "Keeps kernel smem locals in-body as threadgroup storage, refuses a device function's smem local",
    run = mslRewriteSmemLocals
  )
  reg.register("mslThreadCoordinateBuiltins", pkTransform,
    "Threads coordinate builtins down to device functions as parameters," &
    " call sites forward the caller's values",
    run = mslThreadCoordinateBuiltins
  )
  reg.registerInstantiateAddressSpaceGenerics()

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program-scope constants
# ═══════════════════════════════════════════════════════════════════════

proc mslHoistProgramScopeConstants(m: CirModule): string =
  ## Relocates the `{.const_mem.}` locals of every proc into `m.globals` in encounter order,
  ## the declarations dropped from their bodies.
  ##
  ## Contract:
  ## - every proc and scope depth covered, a device-function or nested-scope
  ##   local relocates like a kernel-top-level one
  ## - the initializer is required, a const_mem local without one refuses
  ## - runs after the reserved-word rename, relocated declaration names match
  ##   the renamed body references
  hoistConstMemLocals(m)

# ═══════════════════════════════════════════════════════════════════════
# Device-function `{.smem.}` locals to threadgroup reference parameters
# ═══════════════════════════════════════════════════════════════════════

proc extendedProcType(t: CirType, extra: seq[CirType]): CirType =
  ## A fresh proc type with `extra` appended to the parameters.
  ##
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

proc hoistSmemLocal(fn: CirProc, s: CirStmt, arenas: var seq[SmemArena]) =
  let typ = if s.declType.kind == ctArray:
    tAptr(caThreadgroup, s.declType.aElem)
  else:
    tPtr(caThreadgroup, s.declType)
  fn.params.add(param(s.name, typ))
  arenas.add((s.name, s.declType, typ))

proc hoistSmemLocals(fn: CirProc, arenas: var seq[SmemArena]) =
  ## Hoists the `{.smem.}` locals of one device function into threadgroup
  ## pointer parameters:
  ##
  ## - the declarations drop from the body
  var i = 0
  while i < fn.body.len:
    let s = fn.body[i]
    if s.kind == sVar and s.space == caThreadgroup:
      fn.hoistSmemLocal(s, arenas)
      fn.body.delete i
      continue
    i.inc

proc rewriteArenaReads(e: CirExpr, arenaTypes: Table[string, CirType]) =
  ## Rewrites one expression node that reads a hoisted array arena, in place:
  ##
  ## - arena-symbol references retype to the `aptr` parameter type
  ## - an `at` over the symbol becomes a `pat`, the printed text stays `scratch[i]`
  ##
  ## Expressions are refs, the rewrite applies to every sharer.
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
  ## Retypes the reads of a hoisted array arena across one statement list,
  ## nested bodies included.
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
  ## Appends one argument per hoisted arena of the callee to one call site:
  ##
  ## - a matching-name `{.smem.}` local in the caller supplies the argument,
  ##   an array arena decays to its element pointer in the emitted source,
  ##   an object arena binds by reference
  ## - a kernel caller without one gets the arena synthesized into its body
  ##   as the first statement (the launchDeviceFn wrapper)
  ## - a device-function caller without one refuses, threadgroup storage is
  ##   the caller's to lend
  ##
  ## A callee symbol's proc type grows by the arena types, the growth is
  ## idempotent per shared symbol node and matches the appended arguments.
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
  ## Binds the hoisted-arena arguments of every call in one statement list,
  ## nested bodies included. Returns "" when clean, else the first refusal.
  ##
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
  ## Hoists the `{.smem.}` locals of every device function to threadgroup
  ## reference parameters and binds the call sites.
  ##
  ## - the caller owns the arena: a kernel without a matching local gets one
  ##   synthesized into its body, a device-function caller without one refuses
  ## - runs before `mslRewriteSmemLocals`, which keeps the kernel-owned arenas
  ##   in-body
  var hoisted = initTable[string, seq[SmemArena]]()
  for fn in m.procs:
    if fn.isKernel:
      continue
    var arenas: seq[SmemArena]
    fn.hoistSmemLocals(arenas)
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
  ## Collects the coordinate builtin kind one expression node reads.
  if e.kind == eSym and e.coordBuiltin != cbkNone:
    kinds.incl(e.coordBuiltin)

proc collectCoordReadsStmts(ss: seq[CirStmt]): HashSet[CoordBuiltinKind] =
  ## Collects the coordinate builtin kinds read across one statement list,
  ## nested bodies included.
  var kinds: HashSet[CoordBuiltinKind]
  let collect = proc(e: CirExpr) =
    e.collectCoordReads(kinds)
  ss.walkStmts(nil, collect)
  return kinds

proc builtinParamType(kind: CoordBuiltinKind): CirType =
  ## Parameter type standing in for one coordinate builtin:
  ##
  ## - an `int3` for the vector coordinates: the builtins carry the canonical
  ##   `ivec3` type and the printer emits an `int3` constructor for whole-value reads
  ## - an `int` for the flat thread index
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

proc sortedCoordKinds(kinds: HashSet[CoordBuiltinKind]): seq[CoordBuiltinKind] =
  ## Needed kinds in declaration order, the deterministic parameter order.
  for kind in CoordBuiltinKind:
    if kind != cbkNone and kind in kinds:
      result.add(kind)

proc appendCoordParam(fn: CirProc, kind: CoordBuiltinKind): string =
  ## Appends the parameter standing in for one coordinate builtin to a device function.
  ## A parameter takes the builtin's canonical name so the body's
  ## builtin-marked reads print that name unchanged, in a kernel the name
  ## binds to the attributed parameter, in a device function to this parameter.
  let name = CoordBuiltinCanonicalName[kind]
  for p in fn.params:
    if p.name == name:
      return "at " & fn.name & ": parameter " & name &
        " is named like the coordinate builtin the function needs," &
        " the call sites cannot supply both"
  fn.params.add(param(name, builtinParamType(kind)))
  return ""

proc appendCoordCallArgs(fn: CirProc, calleeProc: CirProc, callee: CirExpr,
                         args: var seq[CirExpr],
                         needs: Table[string, HashSet[CoordBuiltinKind]]): string =
  ## Appends the builtin arguments of one call site, one argument per needed
  ## builtin in the callee's parameter order, and grows the callee symbol's
  ## proc type to match.
  ##
  ## - a kernel forwards the builtin-marked symbol: the printer then emits
  ##   the builtin read, the kernel signature gains the attributed parameter
  ## - a device function forwards its own parameter, the printer emits the bare name
  if callee.kind != eSym or not needs.hasKey(callee.name):
    return ""
  let calleeKinds = sortedCoordKinds(needs[callee.name])
  if callee.typ.kind == ctProc and
      callee.typ.params.len == calleeProc.params.len - calleeKinds.len:
    # one extension per callee symbol, call sites can share the node
    var extra: seq[CirType]
    for kind in calleeKinds:
      extra.add(builtinParamType(kind))
    callee.typ = extendedProcType(callee.typ, extra)
  for kind in calleeKinds:
    if fn.isKernel:
      args.add(sym(CoordBuiltinCanonicalName[kind], builtinParamType(kind),
        builtin = true, coordBuiltin = kind))
    else:
      args.add(sym(CoordBuiltinCanonicalName[kind], builtinParamType(kind)))
  return ""

proc appendCoordForwardingArgsStmts(fn: CirProc, ss: seq[CirStmt],
                                    procs: Table[string, CirProc],
                                    needs: Table[string, HashSet[CoordBuiltinKind]]): string =
  ## Appends the builtin arguments of every call site in one statement list,
  ## calls nested inside expressions included.
  ## Returns "" when clean, else the first refusal.
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
  ## Threads the coordinate builtins down to the device functions as parameters.
  ##
  ## - every device function gains one parameter per transitively needed coordinate builtin in builtin-kind declaration order
  ## - a kernel body is untouched, it reads the builtins it forwards directly
  ## - Metal only: the builtins stay legal reads inside device functions on other targets, so the pass is not registered there
  var needs = initTable[string, HashSet[CoordBuiltinKind]]()
  var direct = initTable[string, HashSet[CoordBuiltinKind]]()
  var callTargets = initTable[string, HashSet[string]]()
  for fn in m.procs:
    needs[fn.name] = initHashSet[CoordBuiltinKind]()
    direct[fn.name] = fn.body.collectCoordReadsStmts()
    var targets: HashSet[string]
    fn.body.collectCallTargets(targets)
    callTargets[fn.name] = targets
  # fixpoint over the call graph: a builtin needed deep in a callee chain reaches every caller, the round count bounds cyclic call graphs
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
    for kind in sortedCoordKinds(needs[fn.name]):
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
  ## Rewrites the `{.smem.}` locals of every proc, nested bodies included:
  ##
  ## - a kernel keeps the local in its body, top-level or nested: the printer
  ##   emits kernel-scope `threadgroup` storage, allocated per threadgroup
  ##   by Metal, no launch-time staging and no parameter
  ## - a device function's smem local refuses, only a kernel owns
  ##   threadgroup storage
  ## - the hoist drains the locals it covers, this walk catches a survivor
  ##   at any depth
  ## - an initializer on a `{.smem.}` local is discarded
  ##
  ## - the pass runs after the reserved-word rename: the kept locals carry
  ##   the renamed names their body already references
  for fn in m.procs:
    var err = ""
    let kernel = fn.isKernel
    let procName = fn.name
    let onStmt = proc(s: CirStmt) =
      if s.kind != sVar or s.space != caThreadgroup:
        return
      if kernel:
        s.init = nil
      elif err.len == 0:
        err = "at " & procName & ": smem local " & s.name &
          " in a device function, only a kernel allocates" &
          " threadgroup storage"
    fn.body.walkStmts(onStmt, nil)
    if err.len > 0:
      return err
  return ""
