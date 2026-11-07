# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ############################################################
#
#            C-family lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSmemInitRewrite*(reg: var PassRegistry) =
  reg.register("smemInitRewrite", pkTransform,
    "Rewrites smem initializers into a bare declaration plus" &
    " element- or member-wise stores",
    run = rewriteSmemInits
  )

# ═══════════════════════════════════════════════════════════════════════
#  Program-scope hoisting
# ═══════════════════════════════════════════════════════════════════════

proc hoistProgramScopeLocalsCore(m: CirModule) =
  ## Relocates the `{.smem.}` and `{.const_mem.}` locals of every proc into
  ## `m.globals`, in encounter order, dropping them from their bodies.
  ##
  ## Contract:
  ## - the smem relocation covers top-level kernel-body locals only, a nested
  ##   or device-function one remains in place and the printer refuses it
  ## - the initializer rewrite ran earlier, a relocated smem local has no initializer
  ## - the const_mem relocation covers every proc at its encounter position,
  ##   a const_mem initializer is required and prints from the relocated node
  ## - runs after the reserved-word rename, relocated declarations use
  ##   the renamed identifiers that their bodies already reference
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.addressSpace == caThreadgroup:
        if not fn.isGlobalFn:
          # a device-function smem local remains in place and the printer refuses it,
          # program-scope workgroup storage belongs to the entrypoint
          i.inc
          continue
        m.globals.add s
        fn.body.delete i
      elif s.kind == sVar and s.addressSpace == caConstant:
        if s.init == nil:
          raiseAssert("const_mem local " & s.name & " without an initializer")
        m.globals.add s
        fn.body.delete i
      else:
        case s.kind
        of sIte:
          hoistConstMemStmts(s.thenBody, m, fn)
          hoistConstMemStmts(s.elseBody, m, fn)
        of sWhile:
          hoistConstMemStmts(s.whileBody, m, fn)
        else:
          discard
        i.inc

template registerCfamilyHoistProgramScopeLocals*(reg: var PassRegistry, passName: string, description: string) =
  reg.register(passName, pkTransform, description,
    run = proc(m: CirModule) {.nimcall.} =
      hoistProgramScopeLocalsCore(m)
  )

# ═══════════════════════════════════════════════════════════════════════
#  Smem initializer rewrite
# ═══════════════════════════════════════════════════════════════════════

proc buildSmemInitStores(name: string, declType: CirType, init: CirExpr): seq[CirStmt] =
  let base = sym(name, declType)
  if init.kind == eAconstr and declType.kind == ctArray:
    for i, el in init.elems:
      result.add asgnS(patE(base, intLit(i, tInt(32))), el)
  elif init.kind == eOconstr and declType.kind == ctStruct:
    for kv in init.kvFields:
      result.add asgnS(dotE(base, kv.name), kv.value)
  else:
    result.add asgnS(base, init)

proc splitSmemInits(ss: var seq[CirStmt]) =
  var i = 0
  while i < ss.len:
    let s = ss[i]
    if s.kind == sVar and s.addressSpace == caThreadgroup and s.init != nil:
      if s.declType.kind in {ctArray, ctStruct} and
          s.init.kind notin {eAconstr, eOconstr}:
        raiseAssert("smem local " & s.name &
          " with an initializer that is not a constructor, no" &
          " element-wise store form exists")
      let stores = buildSmemInitStores(s.name, s.declType, s.init)
      s.init = nil
      var grown: seq[CirStmt]
      grown.add ss[0 .. i]
      grown.add stores
      grown.add ss[i + 1 .. ^1]
      ss = grown
      i.inc stores.len
    elif s.kind == sIte:
      splitSmemInits(s.thenBody)
      splitSmemInits(s.elseBody)
    elif s.kind == sWhile:
      splitSmemInits(s.whileBody)
    i.inc

proc rewriteSmemInits(m: CirModule) =
  ## Kernel-body storage:
  ##
  ## - a local at any depth: initializer-free declaration plus stores
  ##   at the declaration position. The store form is legal runtime storage where the initializer form is not
  ## - Metal drops threadgroup array initializers at runtime
  ## - OpenCL C and CUDA C++ and GLSL forbid the storage-class initializers
  ## - WGSL workgroup variables are zero-initialized
  ##
  ## Untouched and program scope:
  ##
  ## - a device function's smem local: its storage belongs to the caller,
  ##   the ownership refusals stay in charge
  ## - a program-scope `{.smem.}` var that the frontend's direct module path puts
  ##   in `m.globals`: the initializer strips, stores insert at every kernel's entry
  ## - program-scope workgroup storage is fresh per workgroup, so every kernel
  ##   observes the declared values
  for fn in mitems(m.procs):
    if fn.isGlobalFn:
      splitSmemInits(fn.body)
  var entryStores: seq[CirStmt]
  for g in mitems(m.globals):
    if g.kind == sVar and g.addressSpace == caThreadgroup and g.init != nil:
      entryStores.add buildSmemInitStores(g.name, g.declType, g.init)
      g.init = nil
  if entryStores.len > 0:
    for fn in mitems(m.procs):
      if fn.isGlobalFn:
        var grown: seq[CirStmt]
        grown.add entryStores
        grown.add fn.body
        fn.body = grown

