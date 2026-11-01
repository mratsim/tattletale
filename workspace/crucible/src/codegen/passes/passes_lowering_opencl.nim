# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## OpenCL lowering: per-target transforms for the OpenCL C storage model,
## registered by the OpenCL pipeline and run before printing.
##
##   kernel body → decomposeArrayAssign → openclHoistProgramScopeConstants
##   → printed OpenCL C source
##
## - `openclHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into
##   `CirModule.globals` in encounter order, every proc and scope depth covered,
##   the printer emits the relocated declarations as program-scope
##   `__constant` between the typedefs and the forward declarations
## - program-scope `__constant` storage is visible to all functions, a device
##   function reads the constants of the kernels it serves
##
## `{.smem.}` locals are untouched, a mechanical `__local` kernel-body
## declaration the printer emits directly.
##
## Refusal homes:
## - a const_mem local without an initializer refuses in the pass, the printer
##   keeps a nil-init shape guard on the relocated declarations
## - a const_mem local surviving outside the relocation refuses at the printer,
##   the walk covers every proc and scope

import std/tables
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./passes_low_pointers
import ./pass_helpers
import ./passes_low_address_spaces
import ./passes_low_array_assign

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerOpenclLoweringPasses*(reg: var PassRegistry) =
  ## Registers the OpenCL per-target lowering passes:
  ##
  ## - `decomposeArrayAssign`: whole-array copies that OpenCL C cannot express
  ## - the struct-argument explosion runs here, the engine binds one buffer
  ##   per pointer leaf in field order
  ## - `openclHoistProgramScopeConstants`, before printing
  ##
  ## Registration contract: the pass sits after the reserved-word rename,
  ## relocated declaration names match the renamed body references.
  reg.registerDecomposeArrayAssign()
  registerExplodeStructWithPtrFields(reg)
  reg.register("openclHoistProgramScopeConstants", pkTransform,
    "Relocates const_mem locals into the module's program scope",
    run = openclHoistProgramScopeConstants
  )
  reg.register("lowerByrefParams", pkTransform,
    "Passes large structs to device functions as hidden thread pointers",
    run = openclLowerByrefParams
  )
  reg.registerInstantiateAddressSpaceGenerics()

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program-scope constants
# ═══════════════════════════════════════════════════════════════════════

proc openclHoistProgramScopeConstants(m: CirModule): string =
  ## Relocates the `{.const_mem.}` locals of every proc into `m.globals`.
  ## Encounter order, the declarations dropped from their bodies.
  ##
  ## Contract:
  ## - every proc and scope depth covered, a device-function or nested-scope
  ##   local relocates like a kernel-top-level one
  ## - the initializer is required, a const_mem local without one refuses
  ## - runs after the reserved-word rename, relocated declaration names match
  ##   the renamed body references
  hoistConstMemLocals(m)


# ═══════════════════════════════════════════════════════════════════════
# Large-struct params as hidden pointers
# ═══════════════════════════════════════════════════════════════════════

func cirSizeBytes(t: CirType): int =
  ## Byte size of a CIRv2 type, the C layout:
  ## scalars from their bits, arrays elementwise, structs fieldwise,
  ## pointers one machine word.
  case t.kind
  of ctInt, ctUint, ctFloat:
    result = if t.bits > 0: t.bits div 8 else: 8
  of ctBool:
    result = 1
  of ctPtr, ctAptr:
    result = 8
  of ctArray:
    result = t.aN * cirSizeBytes(t.aElem)
  of ctVector:
    result = t.vN * cirSizeBytes(t.vElem)
  of ctStruct:
    result = 0
    for f in t.fields:
      result += cirSizeBytes(f.typ)
  of ctVoid, ctProc, ctSpan:
    result = 0

func isLargeStruct(t: CirType): bool =
  ## True when the struct passes by hidden const reference: the threshold is 24 bytes, the C backend's 3-pointer rule.
  t != nil and t.kind == ctStruct and cirSizeBytes(t) >= 24

proc wrapParamReads(e: CirExpr, oldName: string, ptrSym: CirExpr) =
  ## Rewrites every read of the by-value param `oldName` into a deref of the hidden pointer `ptrSym`, in place over the expression tree.
  ## Expressions are refs, the rewrite applies to the node every sharer sees.
  ##
  ## The rewritten read splices to `deref(ptrSym)` and the walk descends
  ## into the spliced node's children: `ptrSym` names the hidden pointer
  ## and never matches `oldName`.
  if e.isNil:
    return
  proc rewrite(e: CirExpr) =
    if e.kind == eSym and e.name == oldName:
      e.kind = eDeref
      e.derefOperand = ptrSym
  e.walkExprs(rewrite)

proc addrWrapByrefArgs(e: CirExpr, pre: var seq[CirStmt], tmpCount: var int) =
  ## Wraps a call argument in an address when the formal travels as a hidden
  ## pointer and the argument is still the struct value.
  ##
  ## A non-lvalue argument (a call result, a constructor) copies first into
  ## a fresh local: `&` of a temporary is invalid C, the copy is an lvalue.
  ##
  ## - the hoisted declarations run in front of the statement
  ## - nested calls hoist first, every reference reads a computed temp
  ##
  ## The descent goes through `getExprChildren` and the wrap logic reads
  ## the children after it: a call node sees its arguments already rewritten.
  if e.isNil:
    return
  for ch in getExprChildren(e):
    addrWrapByrefArgs(ch, pre, tmpCount)
  if e.kind != eCall:
    return
  let fnTy = e.callee.typ
  if fnTy != nil and fnTy.kind == ctProc:
    for i in 0 ..< e.args.len:
      if i >= fnTy.params.len:
        break
      let formal = fnTy.params[i]
      if formal != nil and formal.kind in {ctPtr, ctAptr} and
          formal.pElem != nil and formal.pElem.kind == ctStruct:
        let at = getExprType(e.args[i])
        if at != nil and at.kind == ctStruct:
          if isLvalue(e.args[i]):
            e.args[i] = addrE(formal, e.args[i])
          else:
            inc tmpCount
            let tmpName = "byref_tmp" & $tmpCount
            let tmp = sym(tmpName, at)
            pre.add(varD(tmpName, at))
            pre.add(asgnS(tmp, e.args[i]))
            e.args[i] = addrE(formal, tmp)

proc addrWrapByrefStmts(stmts: var seq[CirStmt], tmpCount: var int)
  ## Forward declaration: nested statement lists recurse into each other.

proc addrWrapByrefStmt(s: CirStmt, pre: var seq[CirStmt], tmpCount: var int) =
  ## Applies the call-site address wrap over one statement.
  case s.kind
  of sAsgn:
    addrWrapByrefArgs(s.rhs, pre, tmpCount)
  of sIte:
    addrWrapByrefArgs(s.cond, pre, tmpCount)
    addrWrapByrefStmts(s.thenBody, tmpCount)
    addrWrapByrefStmts(s.elseBody, tmpCount)
  of sWhile:
    addrWrapByrefArgs(s.whileCond, pre, tmpCount)
    addrWrapByrefStmts(s.whileBody, tmpCount)
  of sRet:
    addrWrapByrefArgs(s.retVal, pre, tmpCount)
  of sDiscard:
    addrWrapByrefArgs(s.discardVal, pre, tmpCount)
  of sCall:
    let callE = CirExpr(kind: eCall)
    callE.callee = s.callCallee
    callE.args = s.callArgs
    addrWrapByrefArgs(callE, pre, tmpCount)
    s.callCallee = callE.callee
    s.callArgs = callE.args
  of sVar, sConst:
    addrWrapByrefArgs(s.init, pre, tmpCount)
  of sAsm, sEmit:
    discard

proc addrWrapByrefStmts(stmts: var seq[CirStmt], tmpCount: var int) =
  ## Applies the call-site address wrap over every statement of a body.
  ## A hoisted temp declaration inserts in front of its statement.
  var i = 0
  while i < stmts.len:
    var pre: seq[CirStmt] = @[]
    addrWrapByrefStmt(stmts[i], pre, tmpCount)
    for h in pre:
      stmts.insert(h, i)
      i.inc
    i.inc

proc openclLowerByrefParams(m: CirModule): string =
  ## Rewrites a large struct (24 bytes or more) passed by value to a device
  ## function into a hidden thread pointer.
  ##
  ## - the formal re-types, the body's reads deref
  ## - call sites pass the address
  ## - kernels keep their ABI
  ##
  ## The callee's declaration and every call site share one proc-type object,
  ## a single rewrite updates both sides.
  for fn in m.procs:
    if fn.isKernel:
      continue
    var byref: seq[tuple[oldName: string, ptrSym: CirExpr]] = @[]
    for p in fn.params.mitems:
      if p.typ.kind == ctStruct and isLargeStruct(p.typ):
        let ptrTy = tPtr(caThread, p.typ)
        let oldName = p.name
        p.name = "_p_" & p.name
        p.typ = ptrTy
        byref.add((oldName, sym(p.name, ptrTy)))
    for pair in byref:
      wrapParamReadsProc(fn.body, pair.oldName, pair.ptrSym)
  # call-site callee symbols share the proc type: point each rewritten formal at the pointer there, one write per shared type
  var newFormals = initTable[string, seq[CirType]]()
  for fn in m.procs:
    newFormals[fn.name] = block:
      var ps: seq[CirType] = @[]
      for p in fn.params:
        ps.add(p.typ)
      ps
  for fn in m.procs:
    retargetProcTypeStmts(fn.body, newFormals)
    var tmpCount = 0
    addrWrapByrefStmts(fn.body, tmpCount)
  return ""

proc wrapParamReadsProc(stmts: seq[CirStmt], oldName: string, ptrSym: CirExpr) =
  ## Applies the param-read deref wrap over every statement of a body.
  stmts.walkStmts(nil, proc(e: CirExpr) =
    e.wrapParamReads(oldName, ptrSym))

proc retargetProcTypeSyms(e: CirExpr, newFormals: Table[string, seq[CirType]]) =
  ## Points a call-site callee symbol's proc type at the rewritten formals.
  ## Symbols name no children, one visit per node covers the tree.
  if e.isNil:
    return
  proc retarget(e: CirExpr) =
    if e.kind == eSym and e.typ != nil and e.typ.kind == ctProc and
        newFormals.hasKey(e.name):
      let ps = newFormals[e.name]
      for i in 0 ..< min(ps.len, e.typ.params.len):
        e.typ.params[i] = ps[i]
  e.walkExprs(retarget)

proc retargetProcTypeStmts(stmts: seq[CirStmt], newFormals: Table[string, seq[CirType]]) =
  for st in stmts:
    case st.kind
    of sAsgn:
      retargetProcTypeSyms(st.lhs, newFormals)
      retargetProcTypeSyms(st.rhs, newFormals)
    of sIte:
      retargetProcTypeSyms(st.cond, newFormals)
      retargetProcTypeStmts(st.thenBody, newFormals)
      retargetProcTypeStmts(st.elseBody, newFormals)
    of sWhile:
      retargetProcTypeSyms(st.whileCond, newFormals)
      retargetProcTypeStmts(st.whileBody, newFormals)
    of sRet:
      retargetProcTypeSyms(st.retVal, newFormals)
    of sDiscard:
      retargetProcTypeSyms(st.discardVal, newFormals)
    of sCall:
      retargetProcTypeSyms(st.callCallee, newFormals)
      for a in st.callArgs.mitems:
        retargetProcTypeSyms(a, newFormals)
    of sVar, sConst:
      retargetProcTypeSyms(st.init, newFormals)
    of sAsm, sEmit:
      discard
