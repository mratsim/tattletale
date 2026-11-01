# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Lowering: whole-array assignment for the C-family targets.
##
## C and C-family languages have no whole-array assignment. Two shapes refuse
## to compile:
##
## - `a = b` over two `float[N]` locals
## - a declaration initialized from another array, `float c[N] = b;`
##
##   `a = b` over arrays → decomposeArrayAssign → `__builtin_memcpy(&dst, &src, sizeof(T))`
##
## The pass rewrites both shapes into `__builtin_memcpy(&dst, &src, sizeof(T))` calls over any storage class.
##
## Per-target scoping:
## - registered by the Metal, CUDA, and OpenCL bundles
## - GLSL, HLSL, and WGSL assign arrays natively, nodes surface untouched
## - the SPIR-V encoder emits composite load/store natively
##
## A struct holding an array keeps plain assignment: struct assignment is
## a legal whole-value copy that covers its array members.

import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerDecomposeArrayAssign*(reg: var PassRegistry) =
  reg.register("decomposeArrayAssign", pkTransform,
    "Rewrites whole-array assignment and array-valued declaration init into" &
    " __builtin_memcpy calls",
    run = decomposeArrayAssignPass)

# ═══════════════════════════════════════════════════════════════════════
# Decomposition
# ═══════════════════════════════════════════════════════════════════════

proc memcpyAssign(lhs, rhs: CirExpr, lhsType: CirType): CirStmt =
  ## Address operands type as thread pointers.
  ##
  ## - the printers emit `&` without a space qualifier
  ## - the per-address-space callee instantiation re-spaces the pointer
  ##   arguments to the callee's parameter space
  let callee = sym("__builtin_memcpy",
    tProcT([tPtr(caThread, lhsType), tPtr(caThread, lhsType), tUint(-1)], tVoid()))
  let dst = addrE(tPtr(caThread, lhsType), lhs)
  let src = addrE(tPtr(caThread, lhsType), rhs)
  callS(callee, dst, src, sizeofE(lhsType))

proc decomposeArrayAssignStmts(body: var seq[CirStmt]): string =
  ## An array constructor initializer stays, it prints as a braced init list.
  ## A call initializer or call right-hand side refuses: C-family targets cannot
  ## take the address of a call result, a callee returning an array must write
  ## through a pointer parameter instead.
  var lowered: seq[CirStmt]
  for s in body.mitems:
    case s.kind
    of sVar:
      if s.declType.kind == ctArray and s.init != nil and s.init.kind != eAconstr:
        if s.init.kind == eCall:
          return "array declaration binds a call result, C-family targets" &
            " cannot take the address of a call result; bind the call to a local first"
        lowered.add varD(s.name, s.declType, nil, s.space)
        lowered.add memcpyAssign(sym(s.name, s.declType), s.init, s.declType)
      else:
        lowered.add s
    of sAsgn:
      let lhsType = getExprType(s.lhs)
      if lhsType != nil and lhsType.kind == ctArray:
        if s.rhs.kind == eCall:
          return "whole-array assignment binds a call result, C-family targets" &
            " cannot take the address of a call result; bind the call to a local first"
        if not isLvalue(s.rhs):
          return "whole-array assignment right-hand side is not an lvalue, the" &
            " memcpy form copies through storage"
        lowered.add memcpyAssign(s.lhs, s.rhs, lhsType)
      else:
        lowered.add s
    of sIte:
      var err = decomposeArrayAssignStmts(s.thenBody)
      if err.len > 0:
        return err
      err = decomposeArrayAssignStmts(s.elseBody)
      if err.len > 0:
        return err
      lowered.add s
    of sWhile:
      let err = decomposeArrayAssignStmts(s.whileBody)
      if err.len > 0:
        return err
      lowered.add s
    else:
      lowered.add s
  body = lowered

proc decomposeArrayAssignPass(m: CirModule): string =
  for fn in m.procs:
    let err = decomposeArrayAssignStmts(fn.body)
    if err.len > 0:
      return "in proc " & fn.name & ": " & err
