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
## - `mslRewriteSmemLocals`: a kernel's top-level `{.smem.}` local stays
##   in-body as kernel-scope `threadgroup` storage
## - `mslThreadCoordinateBuiltins`: coordinate builtins thread down to device
##   functions as parameters (Metal rejects them outside kernel functions)
##
## Arena ownership: the caller owns the arena and passes it as the call argument.
## A kernel caller without a matching local gets an arena synthesized
## into its body (the launchDeviceFn wrapper).
##
## Threading: call sites forward the caller's values, a kernel reads the builtins
## it forwards directly.
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
import ./passes_low_kernel_params

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
  ## Order note: the smem hoist runs before the builtin threading.
  ## Each pass appends its parameters and matching call-site arguments last,
  ## keeping both orders consistent.
  reg.registerDecomposeArrayAssign()
  registerExplodeKernelStructs(reg)
  reg.register("mslHoistProgramScopeConstants", pkTransform, phaseMain,
    "Relocates const_mem locals into the module's program scope",
    run = mslHoistProgramScopeConstants
  )
  reg.register("mslHoistDeviceFnSmem", pkTransform, phaseMain,
    "Hoists a device function's smem locals to threadgroup reference parameters," &
    " call sites pass the caller's arena",
    run = mslHoistDeviceFnSmem
  )
  reg.register("mslRewriteSmemLocals", pkTransform, phaseMain,
    "Keeps kernel smem locals in-body as threadgroup storage, refuses a device function's smem local",
    run = mslRewriteSmemLocals
  )
  reg.register("mslThreadCoordinateBuiltins", pkTransform, phaseMain,
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
  ## Call-site callee types are interned: structurally identical signatures
  ## share one `CirType` ref. An in-place extension would leak into every
  ## proc and call site carrying that signature, so each call site replaces
  ## its symbol type with a fresh extended copy.
  result = CirType(kind: ctProc)
  result.params = t.params & extra
  result.ret = t.ret

proc procsByName(m: CirModule): Table[string, CirProc] =
  ## Module procs keyed by name, backing the call-site callee lookup.
  for fn in m.procs:
    result[fn.name] = fn

type
  SmemArena = tuple[name: string, declType: CirType, typ: CirType]
    ## One hoisted `{.smem.}` local:
    ##
    ## - `name`, `declType`: the parameter name and the declared local type
    ## - `typ`: the threadgroup pointer type that replaced the local:
    ##   - an array arena takes `aptr[caThreadgroup, element]`, the body
    ##     reads elements through the pointer
    ##   - an object arena takes `ptr[caThreadgroup, object]`, the body
    ##     reads fields through the reference

proc hoistSmemLocal(fn: CirProc, s: CirStmt, arenas: var seq[SmemArena]) =
  ## Hoists one `{.smem.}` local into a threadgroup pointer parameter:
  ##
  ## - an array arena takes an `aptr` over the element type, the emitted
  ##   parameter is `threadgroup T* name` and the body reads elements
  ##   through the pointer
  ## - an object arena takes a `ptr` over the object type, the emitted
  ##   parameter is `threadgroup T& name` and the body reads fields
  ##   through the reference
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
  ## - top-level locals and one `scope` level deep are covered
  ## - the declarations drop from the body
  var i = 0
  while i < fn.body.len:
    let s = fn.body[i]
    if s.kind == sVar and s.space == caThreadgroup:
      fn.hoistSmemLocal(s, arenas)
      fn.body.delete i
      continue
    if s.kind == sScope:
      var j = 0
      while j < s.scopeBody.len:
        let sub = s.scopeBody[j]
        if sub.kind == sVar and sub.space == caThreadgroup:
          fn.hoistSmemLocal(sub, arenas)
          s.scopeBody.delete j
          continue
        j.inc
    i.inc

proc rewriteArenaReads(e: CirExpr, arenaTypes: Table[string, CirType]) =
  ## Rewrites the reads of a hoisted array arena in place:
  ##
  ## - every reference to the arena symbol retypes to the arena's `aptr`
  ##   parameter type
  ## - an `at` over the arena symbol becomes a `pat` over it, the printed
  ##   text stays `scratch[i]`, the element type derives from the `aptr` base
  ##
  ## Expressions are refs, the rewrite applies to every sharer.
  if e.isNil:
    return
  if e.kind == eSym and arenaTypes.hasKey(e.name):
    e.typ = arenaTypes[e.name]
    return
  if e.kind == eAt and e.atBase.kind == eSym and arenaTypes.hasKey(e.atBase.name):
    let base = e.atBase
    let idx = e.atIndex
    base.typ = arenaTypes[base.name]
    e.kind = ePat
    e.patBase = base
    e.patIndex = idx
    rewriteArenaReads(e.patIndex, arenaTypes)
    return
  for ch in getExprChildren(e):
    rewriteArenaReads(ch, arenaTypes)

proc rewriteArenaReadsStmts(ss: seq[CirStmt], arenaTypes: Table[string, CirType]) =
  ## Statement-walk overload of `rewriteArenaReads`, nested bodies included.
  for s in ss:
    case s.kind
    of sVar, sConst:
      rewriteArenaReads(s.init, arenaTypes)
    of sAsgn:
      rewriteArenaReads(s.lhs, arenaTypes)
      rewriteArenaReads(s.rhs, arenaTypes)
    of sIte:
      rewriteArenaReads(s.cond, arenaTypes)
      rewriteArenaReadsStmts(s.thenBody, arenaTypes)
      rewriteArenaReadsStmts(s.elseBody, arenaTypes)
    of sLoop:
      rewriteArenaReadsStmts(s.loopBody, arenaTypes)
    of sScope:
      rewriteArenaReadsStmts(s.scopeBody, arenaTypes)
    of sAssume:
      rewriteArenaReads(s.assumeCond, arenaTypes)
    of sRet:
      rewriteArenaReads(s.retVal, arenaTypes)
    of sDiscard:
      rewriteArenaReads(s.discardVal, arenaTypes)
    of sCall:
      for a in s.callArgs:
        rewriteArenaReads(a, arenaTypes)
    of sLab, sJmp, sCase, sAsm, sEmit:
      discard

proc ownedSmemNames(fn: CirProc): HashSet[string] =
  ## Names under which a proc owns threadgroup storage:
  ##
  ## - the hoisted arena parameters the proc received
  ## - the `{.smem.}` locals still declared in the body, top level and one
  ##   `scope` level deep
  for p in fn.params:
    if p.typ.kind == ctPtr and p.typ.space == caThreadgroup:
      result.incl(p.name)
  for s in fn.body:
    if s.kind == sVar and s.space == caThreadgroup:
      result.incl(s.name)
    if s.kind == sScope:
      for sub in s.scopeBody:
        if sub.kind == sVar and sub.space == caThreadgroup:
          result.incl(sub.name)

proc bindSmemArgs(fn: CirProc, calleeProc: CirProc, callee: CirExpr,
                  args: var seq[CirExpr],
                  arenas: seq[SmemArena], owned: var HashSet[string]): string =
  ## Appends one argument per hoisted arena of the callee to one call site:
  ##
  ## - the caller owns a matching-name arena: the argument names it, typed
  ##   as the callee's parameter. An array arena decays to its element
  ##   pointer in the emitted source, an object arena binds by reference.
  ## - a kernel caller without one: the arena synthesizes into the kernel
  ##   body as the first statement, so a generated kernel launches a device
  ##   function that owns its scratch (the launchDeviceFn wrapper)
  ## - a device-function caller without one refuses, it has no threadgroup
  ##   storage to lend
  ##
  ## A callee symbol's proc type grows by the arena types. The growth is
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

proc bindSmemCallSiteStmts(fn: CirProc, ss: seq[CirStmt],
                           procs: Table[string, CirProc],
                           hoisted: Table[string, seq[SmemArena]],
                           owned: var HashSet[string]): string =
  ## Binds the hoisted-arena arguments of every call in one statement list,
  ## nested bodies included. Returns "" when clean, else the first refusal.
  ##
  ## Walks a snapshot of the statement references: binding a kernel call site
  ## synthesizes its arena into `fn.body`, and the synthesis must not perturb
  ## the walk (a synthesized declaration binds no call).
  var walked: seq[CirStmt]
  for s in ss:
    walked.add(s)
  for s in walked:
    case s.kind
    of sCall:
      if s.callCallee.kind == eSym and hoisted.hasKey(s.callCallee.name):
        let err = bindSmemArgs(fn, procs[s.callCallee.name], s.callCallee,
          s.callArgs, hoisted[s.callCallee.name], owned)
        if err.len > 0:
          return err
    of sAsgn:
      if s.rhs.kind == eCall and s.rhs.callee.kind == eSym and
          hoisted.hasKey(s.rhs.callee.name):
        let err = bindSmemArgs(fn, procs[s.rhs.callee.name], s.rhs.callee,
          s.rhs.args, hoisted[s.rhs.callee.name], owned)
        if err.len > 0:
          return err
    of sIte:
      let thenErr = bindSmemCallSiteStmts(fn, s.thenBody, procs, hoisted, owned)
      if thenErr.len > 0:
        return thenErr
      let elseErr = bindSmemCallSiteStmts(fn, s.elseBody, procs, hoisted, owned)
      if elseErr.len > 0:
        return elseErr
    of sLoop:
      let loopErr = bindSmemCallSiteStmts(fn, s.loopBody, procs, hoisted, owned)
      if loopErr.len > 0:
        return loopErr
    of sScope:
      let scopeErr = bindSmemCallSiteStmts(fn, s.scopeBody, procs, hoisted, owned)
      if scopeErr.len > 0:
        return scopeErr
    else:
      discard
  return ""

proc mslHoistDeviceFnSmem(m: CirModule): string =
  ## Hoists the `{.smem.}` locals of every device function to threadgroup
  ## reference parameters and binds the call sites.
  ##
  ## Contract:
  ## - the caller owns the arena: a matching-name `{.smem.}` local, or one
  ##   synthesized into a kernel body when the kernel owns none (the launchDeviceFn wrapper case)
  ## - a device-function caller without a matching arena refuses
  ## - a callee symbol's proc type grows by the arena types to match the appended parameters and arguments
  ## - runs before `mslRewriteSmemLocals`, which keeps the kernel-owned
  ##   arenas in-body as threadgroup storage
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
    var owned = fn.ownedSmemNames()
    let err = bindSmemCallSiteStmts(fn, fn.body, procs, hoisted, owned)
    if err.len > 0:
      return err
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins threaded down to device functions
# ═══════════════════════════════════════════════════════════════════════

proc collectCoordReads(e: CirExpr, kinds: var HashSet[CoordBuiltinKind]) =
  ## Collects the coordinate builtin kinds one expression reads across the whole tree.
  if e.isNil:
    return
  if e.kind == eSym and e.coordBuiltin != cbkNone:
    kinds.incl(e.coordBuiltin)
    return
  for ch in getExprChildren(e):
    collectCoordReads(ch, kinds)

proc collectCoordReadsStmts(ss: seq[CirStmt], kinds: var HashSet[CoordBuiltinKind]) =
  ## Statement-walk overload of `collectCoordReads`, nested bodies included.
  for s in ss:
    case s.kind
    of sVar, sConst:
      collectCoordReads(s.init, kinds)
    of sAsgn:
      collectCoordReads(s.lhs, kinds)
      collectCoordReads(s.rhs, kinds)
    of sIte:
      collectCoordReads(s.cond, kinds)
      collectCoordReadsStmts(s.thenBody, kinds)
      collectCoordReadsStmts(s.elseBody, kinds)
    of sLoop:
      collectCoordReadsStmts(s.loopBody, kinds)
    of sScope:
      collectCoordReadsStmts(s.scopeBody, kinds)
    of sAssume:
      collectCoordReads(s.assumeCond, kinds)
    of sRet:
      collectCoordReads(s.retVal, kinds)
    of sDiscard:
      collectCoordReads(s.discardVal, kinds)
    of sCall:
      collectCoordReads(s.callCallee, kinds)
      for a in s.callArgs:
        collectCoordReads(a, kinds)
    of sLab, sJmp, sCase, sAsm, sEmit:
      discard

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
  ## One fixpoint step of the need computation over the precomputed per-proc
  ## reads and call targets. Returns true when the proc's need set grew.
  ## A proc needs every coordinate builtin its body reads plus every builtin
  ## its callees need.
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
  ## - a kernel forwards the builtin-marked symbol: the printer emits the builtin read,
  ##   the kernel signature gains the attributed parameter
  ## - a device function forwards its own parameter, the plain symbol prints
  ##   the bare name
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

proc appendCoordForwardingArgs(fn: CirProc, e: CirExpr, procs: Table[string, CirProc],
                               needs: Table[string, HashSet[CoordBuiltinKind]]): string =
  ## Appends the builtin arguments of every call nested in one expression,
  ## outermost call last. Returns "" when clean, else the first refusal.
  if e.isNil:
    return ""
  if e.kind == eCall:
    for i in 0 ..< e.args.len:
      let err = appendCoordForwardingArgs(fn, e.args[i], procs, needs)
      if err.len > 0:
        return err
    return appendCoordCallArgs(fn, procs.getOrDefault(e.callee.name), e.callee, e.args, needs)
  for ch in getExprChildren(e):
    let err = appendCoordForwardingArgs(fn, ch, procs, needs)
    if err.len > 0:
      return err
  return ""

proc appendCoordForwardingArgsStmts(fn: CirProc, ss: seq[CirStmt],
                                    procs: Table[string, CirProc],
                                    needs: Table[string, HashSet[CoordBuiltinKind]]): string =
  ## Statement-walk overload of `appendCoordForwardingArgs` over nested bodies.
  ## Returns "" when clean, else the first refusal.
  for s in ss:
    case s.kind
    of sVar, sConst:
      let err = appendCoordForwardingArgs(fn, s.init, procs, needs)
      if err.len > 0:
        return err
    of sAsgn:
      let lerr = appendCoordForwardingArgs(fn, s.lhs, procs, needs)
      if lerr.len > 0:
        return lerr
      let rerr = appendCoordForwardingArgs(fn, s.rhs, procs, needs)
      if rerr.len > 0:
        return rerr
    of sIte:
      let cerr = appendCoordForwardingArgs(fn, s.cond, procs, needs)
      if cerr.len > 0:
        return cerr
      let thenErr = appendCoordForwardingArgsStmts(fn, s.thenBody, procs, needs)
      if thenErr.len > 0:
        return thenErr
      let elseErr = appendCoordForwardingArgsStmts(fn, s.elseBody, procs, needs)
      if elseErr.len > 0:
        return elseErr
    of sLoop:
      let loopErr = appendCoordForwardingArgsStmts(fn, s.loopBody, procs, needs)
      if loopErr.len > 0:
        return loopErr
    of sScope:
      let scopeErr = appendCoordForwardingArgsStmts(fn, s.scopeBody, procs, needs)
      if scopeErr.len > 0:
        return scopeErr
    of sAssume:
      let aerr = appendCoordForwardingArgs(fn, s.assumeCond, procs, needs)
      if aerr.len > 0:
        return aerr
    of sRet:
      let rerr = appendCoordForwardingArgs(fn, s.retVal, procs, needs)
      if rerr.len > 0:
        return rerr
    of sDiscard:
      let derr = appendCoordForwardingArgs(fn, s.discardVal, procs, needs)
      if derr.len > 0:
        return derr
    of sCall:
      for a in s.callArgs:
        let aerr = appendCoordForwardingArgs(fn, a, procs, needs)
        if aerr.len > 0:
          return aerr
      let cerr = appendCoordCallArgs(fn, procs.getOrDefault(s.callCallee.name), s.callCallee, s.callArgs, needs)
      if cerr.len > 0:
        return cerr
    of sLab, sJmp, sCase, sAsm, sEmit:
      discard
  return ""

proc mslThreadCoordinateBuiltins(m: CirModule): string =
  ## Threads the coordinate builtins down to the device functions as parameters.
  ##
  ## Contract:
  ## - every device function gains one parameter per transitively needed coordinate builtin in builtin-kind declaration order
  ## - a kernel body is untouched: it reads the builtins it forwards directly,
  ##   the printer emits the attributed parameters from those reads
  ## - every call site forwards the caller's values and the callee symbol's
  ##   proc type grows by the appended parameter types
  ## - Metal only: the builtins stay legal reads inside device functions on other targets, so the pass is not registered there
  var needs = initTable[string, HashSet[CoordBuiltinKind]]()
  var direct = initTable[string, HashSet[CoordBuiltinKind]]()
  var callTargets = initTable[string, HashSet[string]]()
  for fn in m.procs:
    needs[fn.name] = initHashSet[CoordBuiltinKind]()
    var kinds: HashSet[CoordBuiltinKind]
    fn.body.collectCoordReadsStmts(kinds)
    direct[fn.name] = kinds
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
  ## Rewrites the top-level `{.smem.}` locals of every proc:
  ##
  ## - a kernel keeps the local in its body: the printer emits kernel-scope
  ##   `threadgroup` storage that Metal allocates per threadgroup,
  ##   no launch-time staging and no parameter
  ## - a device function's smem local refuses, only a kernel owns threadgroup storage, the hoist covers top-level and one-scope-deep locals
  ## - an initializer on a `{.smem.}` local is discarded
  ##
  ## - the rewrite covers top-level body locals only, a nested one keeps
  ##   its declaration and prints in place
  ## - the pass runs after the reserved-word rename: the kept locals carry
  ##   the renamed names their body already references
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.space == caThreadgroup:
        if fn.isKernel:
          s.init = nil
          i.inc
        else:
          return "at " & fn.name & ": smem local " & s.name &
            " in a device function, only a kernel allocates" &
            " threadgroup storage"
      else:
        i.inc
  return ""
