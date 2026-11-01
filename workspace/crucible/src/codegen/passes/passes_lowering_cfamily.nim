# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Shared C-family lowering: program-scope hoisting.
## One registration template per pass, one registration call per target.
##
## - `registerCfamilyHoistProgramScopeLocals`: registers the smem and const_mem relocation.
## The pointer-local explosion walk lives in passes_low_pointers.
##   `SmemInitPolicy` is the target's policy for an initializer on a smem local.
##   `smemInitRefuse` refuses naming the target's rule.
##   `smemInitEmit` keeps the initializer for the printer to emit.

import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Program-scope hoisting
# ═══════════════════════════════════════════════════════════════════════

type SmemInitPolicy* = enum
  ## Target policy for an initializer on a `{.smem.}` local.
  smemInitRefuse
    ## the target's workgroup storage takes no initializer so the hoist
    ## refuses naming the target's rule
  smemInitEmit
    ## the target's workgroup storage accepts an initializer for the printer to emit

proc hoistProgramScopeLocalsCore(m: CirModule, smemInit: SmemInitPolicy, smemInitReason: string): string =
  ## Relocates the `{.smem.}` and `{.const_mem.}` locals of every proc into
  ## `m.globals`, in encounter order, dropping them from their bodies.
  ##
  ## Contract:
  ## - the smem relocation covers top-level kernel-body locals only, a nested
  ##   or device-function one stays in place and refuses at the printer
  ## - under `smemInitRefuse` a smem local with an initializer refuses,
  ##   `smemInitReason` names the target's rule, under `smemInitEmit` it
  ##   relocates with the local for the printer to emit
  ## - the const_mem relocation covers every proc and scope depth, the shared
  ##   const_mem walk recurses into the nested scopes at their encounter position,
  ##   a const_mem initializer is required, the printer emits it
  ##   from the relocated node
  ## - runs after the reserved-word rename, the relocated declarations use
  ##   the renamed names that their body already references
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.space == caThreadgroup:
        if not fn.isKernel:
          # a device-function smem local stays in place and the printer refuses it,
          # program-scope workgroup storage belongs to the kernel
          i.inc
          continue
        if s.init != nil and smemInit == smemInitRefuse:
          return "at " & fn.name & ": smem local " & s.name &
            " with an initializer, " & smemInitReason
        m.globals.add s
        fn.body.delete i
      elif s.kind == sVar and s.space == caConstant:
        if s.init == nil:
          return "at " & fn.name & ": const_mem local " & s.name &
            " without an initializer"
        m.globals.add s
        fn.body.delete i
      else:
        case s.kind
        of sIte:
          let thenErr = hoistConstMemStmts(m, fn, s.thenBody)
          if thenErr.len > 0:
            return thenErr
          let elseErr = hoistConstMemStmts(m, fn, s.elseBody)
          if elseErr.len > 0:
            return elseErr
        of sLoop:
          let loopErr = hoistConstMemStmts(m, fn, s.loopBody)
          if loopErr.len > 0:
            return loopErr
        else:
          discard
        i.inc
  return ""

template registerCfamilyHoistProgramScopeLocals*(reg: var PassRegistry, passName: string,
    description: string, smemInit: SmemInitPolicy, smemInitReason = "") =
  ## Registers the program-scope hoist for one target.
  ##
  ## Args:
  ## - `passName`: the registry key, one of `glslHoistProgramScopeLocals`
  ##   and `wgslHoistProgramScopeLocals`
  ## - `description`: the registry description
  ## - `smemInit`: the target's policy for an initializer on a smem local
  ## - `smemInitReason`: the target's rule an initializer violates, required
  ##   under `smemInitRefuse`, ignored under `smemInitEmit`
  when smemInit == smemInitRefuse:
    doAssert smemInitReason.len > 0,
      passName & ": smemInitRefuse requires the target's refusal reason"
  reg.register(passName, pkTransform, description,
    run = proc(m: CirModule): string {.nimcall.} =
      hoistProgramScopeLocalsCore(m, smemInit, smemInitReason)
  )
