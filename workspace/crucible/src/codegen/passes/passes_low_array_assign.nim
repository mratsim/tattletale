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

# ############################################################
#
#            Array-assign lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerDecomposeArrayAssign*(reg: var PassRegistry) =
  reg.register("decomposeArrayAssign", pkTransform,
    "Rewrites whole-array assignment, array-valued declaration init, and" &
    " constructor assignment into __builtin_memcpy calls",
    run = decomposeArrayAssignPass)

# ═══════════════════════════════════════════════════════════════════════
#  Decomposition
# ═══════════════════════════════════════════════════════════════════════

proc memcpyAssign(lhs, rhs: CirExpr, lhsType: CirType): CirStmt =
  ## One whole-array copy, `__builtin_memcpy(&lhs, &rhs, sizeof(lhsType))`.
  ##
  ## Both operands are addressed, so `rhs` names storage when this proc runs.
  ## The caller refuses call results and non-lvalues and pre-binds constructor values to fresh locals.
  ##
  ## The pointer operands type as thread pointers. The address-space generic
  ## instantiation re-spaces call arguments to the callee's parameter space,
  ## so one placeholder space covers every source space.
  let callee = sym("__builtin_memcpy",
    tProcT([tPtr(caThread, lhsType), tPtr(caThread, lhsType), tUint(-1)], tVoid()))
  let dst = addrE(tPtr(caThread, lhsType), lhs)
  let src = addrE(tPtr(caThread, lhsType), rhs)
  callS(callee, dst, src, sizeofE(lhsType))

proc bindConstrLocal(rhs: CirExpr, lhsType: CirType, tmpCount: var int): tuple[decl: CirStmt, local: CirExpr] =
  ## Binds a constructor value to one fresh local through its braced init list.
  ## The memcpy copies from that local. A constructor is no lvalue.
  ## C-family targets cannot take its address in place.
  ##
  ## Returns the local's declaration and the local as an expression.
  ## Identifiers count per proc, starting at `memcpy_tmp1`.
  inc tmpCount
  let name = "memcpy_tmp" & $tmpCount
  result.decl = varD(name, lhsType, rhs)
  result.local = sym(name, lhsType)

proc decomposeArrayAssignStmts(body: var seq[CirStmt], tmpCount: var int): string =
  ## Rewrites the whole-array copies of one statement sequence in place.
  ##
  ## Returns "" on success, else the refusal text naming the first shape
  ## the pass cannot lower:
  ##
  ## - a declaration or assignment binding a call result: C-family targets
  ##   cannot take the address of a call result, bind the call to a local first
  ## - an assignment right side that is neither an lvalue nor a constructor:
  ##   the memcpy form copies through storage
  ##
  ## `tmpCount` numbers the fresh constructor locals across the sequence,
  ## counting continues through nested branch and loop bodies.
  var lowered: seq[CirStmt]
  for s in body.mitems:
    case s.kind
    of sVar:
      if s.declType.kind == ctArray and s.init != nil and s.init.kind != eAconstr:
        if s.init.kind == eCall:
          return "array declaration binds a call result, C-family targets" &
            " cannot take the address of a call result; bind the call to a local first"
        lowered.add varD(s.name, s.declType, nil, s.addressSpace)
        lowered.add memcpyAssign(sym(s.name, s.declType), s.init, s.declType)
      else:
        lowered.add s
    of sAsgn:
      let lhsType = getExprType(s.lhs)
      if lhsType != nil and lhsType.kind == ctArray:
        if s.rhs.kind == eCall:
          return "whole-array assignment binds a call result, C-family targets" &
            " cannot take the address of a call result; bind the call to a local first"
        if s.rhs.kind notin {eAconstr, eOconstr} and not isLvalue(s.rhs):
          return "whole-array assignment right-hand side is not an lvalue, the" &
            " memcpy form copies through storage"
        if s.rhs.kind in {eAconstr, eOconstr}:
          let bound = bindConstrLocal(s.rhs, lhsType, tmpCount)
          lowered.add bound.decl
          lowered.add memcpyAssign(s.lhs, bound.local, lhsType)
        else:
          lowered.add memcpyAssign(s.lhs, s.rhs, lhsType)
      else:
        lowered.add s
    of sIte:
      var err = decomposeArrayAssignStmts(s.thenBody, tmpCount)
      if err.len > 0:
        return err
      err = decomposeArrayAssignStmts(s.elseBody, tmpCount)
      if err.len > 0:
        return err
      lowered.add s
    of sWhile:
      let err = decomposeArrayAssignStmts(s.whileBody, tmpCount)
      if err.len > 0:
        return err
      lowered.add s
    else:
      lowered.add s
  body = lowered

proc decomposeArrayAssignPass(m: CirModule): string =
  for fn in m.procs:
    var tmpCount = 0
    let err = decomposeArrayAssignStmts(fn.body, tmpCount)
    if err.len > 0:
      return "in proc " & fn.name & ": " & err
