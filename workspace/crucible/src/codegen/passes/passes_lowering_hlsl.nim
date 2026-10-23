# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            HLSL lowering passes
#
# ############################################################
#
# Lower the crucible IR onto the glslang HLSL frontend's storage model,
# registered by the HLSL pipeline only. The chain matches the GLSL bundle
# minus two passes, the HLSL resource model has no use for them:
# - insertPointerDeref: the printer reads through gpuDeref on index bases,
#   explicit derefs are unnecessary IR
# - qualifyVarParams: `var` params are rejected by the HLSL validation pass,
#   there is nothing to qualify
#
# inlineKernelCallees registers last, after every pass that can create calls.
# The glslang HLSL frontend rejects buffer-resource-taking device function parameters,
# so calls to such callees cannot compile.
# The pass splices every callee reachable from a kernel body into that body.

import std / [macros, sets, sequtils, tables]
import ../../builtins/builtins_functions
import ../ir/ast_datatypes
import ./pass_datatypes
import ./pass_helpers
import ./pass_registry
import ./passes_lowering
import ./passes_low_views
import ./passes_low_array_assign
import ./passes_low_address_spaces

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerInlineKernelCallees*(reg: var PassRegistry) =
  reg.register("inlineKernelCallees", pkTransform, phaseMain,
    "Splices every non-intrinsic callee reachable from a kernel body into the kernel body, dead fnTab entries drop. " &
    "Refuses: early returns, calls in loop headers or while conditions, recursion, " &
    "dereferences of pointer parameters bound to a pointer value, and references to names the kernel declares (the name-based splice would capture them)",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.inlineKernelCallees()
  )

func registerHlslLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = false)
  reg.registerSharedLoweringPasses()
  reg.registerDecomposeArrayAssign()
  reg.registerRenameReservedIdentifiers("hlslReservedIdentifiers", HlslReserved)
  reg.registerInstantiateImplicitAddressSpaceGenerics()
  # Runs after every pass that can create calls and before the HLSL validations, so kernels reach the printer without device function calls.
  reg.registerInlineKernelCallees()

const HlslReserved = [
  # statements and declarations
  "if", "else", "for", "while", "do", "switch", "case", "default",
  "break", "continue", "return", "discard", "struct", "class", "interface",
  "typedef", "namespace", "using", "cbuffer", "tbuffer", "register",
  "packoffset", "technique", "pass", "compile", "main",
  # storage qualifiers and modifiers
  "static", "const", "extern", "uniform", "volatile", "shared", "groupshared",
  "inline", "in", "out", "inout", "precise", "row_major", "column_major",
  # builtin scalar, vector and matrix type names
  "void", "bool", "int", "uint", "float", "double", "half", "fixed", "string",
  "int1", "int2", "int3", "int4", "uint1", "uint2", "uint3", "uint4",
  "float1", "float2", "float3", "float4", "bool1", "bool2", "bool3", "bool4",
  "half2", "half3", "half4", "double1", "double2", "double3", "double4",
  "matrix", "float2x2", "float3x3", "float4x4", "int2x2", "int3x3", "int4x4",
  "uint2x2", "uint3x3", "uint4x4", "bool2x2", "bool3x3", "bool4x4",
  # fixed-width type names: int64_t and uint64_t are rejected by validation,
  # the remaining stdint names are plain parse errors in the HLSL frontend
  "int32_t", "uint32_t", "float32_t", "float64_t", "int64_t", "uint64_t",
  # resource types the printer emits and their siblings
  "ConstantBuffer", "TextureBuffer", "StructuredBuffer", "RWStructuredBuffer",
  "AppendStructuredBuffer", "ConsumeStructuredBuffer", "ByteAddressBuffer",
  "RWByteAddressBuffer", "Texture1D", "Texture2D", "Texture3D", "TextureCube",
  "RWTexture1D", "RWTexture2D", "RWTexture3D",
  "SamplerState", "SamplerComparisonState",
  # semantics names the printer emits and their siblings
  "numthreads", "SV_DispatchThreadID", "SV_GroupID", "SV_GroupThreadID",
  "SV_GroupIndex", "SV_Position", "SV_NumWorkgroups", "SV_Target",
  # literals
  "true", "false",
].toHashSet()

# ═══════════════════════════════════════════════════════════════════════
# Kernel callee inlining
# ═══════════════════════════════════════════════════════════════════════
#
# Device function calls cannot compile through the glslang HLSL frontend (verified with glslang 16.6).
# StructuredBuffer parameters cannot cross function signatures, buffer-taking device functions segfault outright.
#
# inlineKernelCallees splices every non-intrinsic callee reachable from a kernel into the kernel body, the printer never sees a device call.
#
# Per call site:
#
# - a pointer parameter's uses bind to the target its argument addresses,
#   a dereference of a parameter bound to a pointer value instead refuses
# - a by-value parameter binds once to a fresh copy seeded from its argument,
#   the callee's writes hit the copy
# - a by-reference parameter (large structs) substitutes its argument at each use
#
# Splice scoping:
#
# - every callee-owned name renames with a fresh suffix, the same callee
#   spliced twice keeps distinct locals
# - the implicit `result` of a non-void callee routes to the caller's target,
#   a var initializer, an assignment target, or a fresh hoisted temp
# - the trailing return drops (void) or stores into the target (non-void)
#
# Expression-position calls hoist onto a fresh temp declared just before the statement.
# A call inside a loop header or a while condition refuses loudly, those positions re-evaluate per iteration.
# An early return refuses too, it would exit mid-splice.
#
# Reference matching inside the spliced body is name-based only, symbol identity is per-body.
# The callee's `result` and its parameter references share no Symbol objects with the kernel's nodes.
# Every callee-owned name renames, so surviving references resolve outside the callee names.
# A surviving reference to a name the kernel declares refuses, the message names the capture.

type KernelSplicer = object
  counter: int                ## fresh local suffix source, unique per kernel
  usedNames: HashSet[string]  ## every name the kernel body already uses, fresh names avoid them
  inlineStack: seq[string]    ## iSyms of the callees being spliced, recursion guard

proc freshName(st: var KernelSplicer, base: string): string =
  ## Fresh local name: `base` suffixed with the splice counter, skipping names the kernel body already uses
  while true:
    result = base & "_" & $st.counter
    inc st.counter
    if result notin st.usedNames:
      break
  st.usedNames.incl result

proc collectLocalDecls(n: GpuAst, names: var seq[string], types: var Table[string, GpuType]) =
  ## Collects the declared local names of a device function body in order,
  ## covering `var` declarations, `constexpr` identifiers and loop variables.
  ##
  ## - `names` receives one entry per local declaration
  ## - `types` maps each name to its declared type
  ##
  ## A name redeclared in disjoint branches merges when both declarations
  ## carry the same type, the branch scopes keep them apart in the output
  case n.kind
  of gpuVar:
    let name = n.vName.symbol.name
    if name in types:
      doAssert types[name] == n.vType,
        "device function body declares `" & name & "` twice with different types, name-based splicing is ambiguous"
    else:
      names.add name
      types[name] = n.vType
    collectLocalDecls(n.vInit, names, types)
  of gpuConstexpr:
    let name = n.cIdent.symbol.name
    if name in types:
      doAssert types[name] == n.cType,
        "device function body declares `" & name & "` twice with different types, name-based splicing is ambiguous"
    else:
      names.add name
      types[name] = n.cType
    collectLocalDecls(n.cValue, names, types)
  of gpuFor:
    let name = n.fVar.symbol.name
    if name in types:
      doAssert types[name] == n.fVar.symbol.typ,
        "device function body declares `" & name & "` twice with different types, name-based splicing is ambiguous"
    else:
      names.add name
      types[name] = n.fVar.symbol.typ
    collectLocalDecls(n.fBody, names, types)
  of gpuBlock:
    for s in n.statements:
      collectLocalDecls(s, names, types)
  of gpuIf:
    collectLocalDecls(n.ifThen, names, types)
    for el in n.ifElifs:
      collectLocalDecls(el.body, names, types)
    if n.ifElse.kind != gpuDiscard:
      collectLocalDecls(n.ifElse, names, types)
  of gpuWhile:
    collectLocalDecls(n.wBody, names, types)
  else:
    discard

proc isSpliceableCall(ctx: GpuContext, call: GpuAst): bool =
  ## True when a call's callee has an IR body the printer cannot emit, so the call splices into the caller.
  ## Materialized numeric builtins (max/min/abs) print the backend intrinsic, barriers, reductions and atomics emit as plain calls
  if call.cName.symbol.isNil:
    return false
  (call.cName in ctx.allFnTab or call.cName in ctx.genericInsts) and
    numericBuiltinDisplayName(call.cName.symbol.name).len == 0

proc containsSpliceableCall(ctx: GpuContext, n: GpuAst): bool =
  ## True when an expression subtree calls a device function: loop headers and while conditions evaluate per iteration.
  ## Intrinsic calls print inline and never trip this check
  case n.kind
  of gpuCall:
    if ctx.isSpliceableCall(n):
      return true
    for arg in n.cArgs:
      if ctx.containsSpliceableCall(arg):
        return true
    false
  else:
    for ch in n:
      if ctx.containsSpliceableCall(ch):
        return true
    false

type SpliceEnv = object
  subst: Table[string, GpuAst]   ## by-value parameter: the fresh copy its argument seeds
  ptrBind: Table[string, GpuAst] ## pointer parameter: the target its argument addresses
  ptrValues: HashSet[string]     ## pointer parameters bound to a pointer value, a dereference refuses
  renames: Table[string, Symbol] ## callee-owned local: its fresh symbol
  retName: string                ## the callee's implicit `result` name, empty for void callees
  retBind: GpuAst                ## the caller's target for the callee's result, nil for void callees
  usedNames: HashSet[string]     ## names the kernel already uses, a surviving reference to one is a capture
  calleeName: string             ## the callee's name, for refusal messages

proc spliceRewrite(n: var GpuAst, env: SpliceEnv) =
  ## One pass over a cloned device function body:
  ##
  ## Binding:
  ##
  ## - by-value parameters substitute by their fresh copies
  ## - pointer parameters substitute by their bound targets
  ## - the implicit `result` bindings route to the caller's target
  ##
  ## Splice:
  ##
  ## - callee-owned locals rename to their fresh symbols
  ## - returns lower to drops or result stores
  ## - a surviving reference to a name the kernel declares refuses, the name-based splice would capture the kernel's binding
  ##
  ## Pointer parameters substitute at use sites: a bare use reads the bound pointer value and a dereference reads the bound target.
  case n.kind
  of gpuIdent:
    let name = n.symbol.name
    if name in env.subst:
      n = env.subst[name].clone()
    elif env.retName.len > 0 and name == env.retName:
      n = env.retBind.clone()
    elif name in env.renames:
      n = GpuAst(kind: gpuIdent, symbol: env.renames[name])
    elif name in env.ptrBind:
      n = env.ptrBind[name].clone()
    elif name in env.usedNames and n.symbol.symKind != gsBuiltin:
      # a builtin reads the same hardware value everywhere, there is no kernel binding to capture
      error "HLSL: device function `" & env.calleeName & "` references `" & name &
        "` which the kernel declares, the name-based splice would capture the kernel's binding"
  of gpuVar:
    if n.vName.symbol.name == env.retName:
      # the caller's target decl already exists, the callee's own decl drops
      n = GpuAst(kind: gpuDiscard)
    else:
      if n.vName.symbol.name in env.renames:
        n.vName = GpuAst(kind: gpuIdent, symbol: env.renames[n.vName.symbol.name])
      spliceRewrite(n.vInit, env)
  of gpuReturn:
    if n.rValue.kind == gpuDiscard:
      n = GpuAst(kind: gpuDiscard)
    else:
      if env.retName.len == 0:
        error "HLSL: void device function returns a value"
      if n.rValue.kind == gpuIdent and n.rValue.symbol.name == env.retName:
        # the target already holds the result, the trailing return drops
        n = GpuAst(kind: gpuDiscard)
      else:
        spliceRewrite(n.rValue, env)
        n = GpuAst(kind: gpuAssign, aLeft: env.retBind.clone(), aRight: n.rValue)
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol.name in env.ptrBind:
      if n.dOf.symbol.name in env.ptrValues:
        error "HLSL: device function `" & env.calleeName & "` dereferences pointer parameter `" &
          n.dOf.symbol.name & "` bound to a pointer value, cannot inline, only an address argument binds the dereference to its target"
      n = env.ptrBind[n.dOf.symbol.name].clone()
    else:
      spliceRewrite(n.dOf, env)
  of gpuFor:
    if n.fVar.symbol.name in env.renames:
      n.fVar = GpuAst(kind: gpuIdent, symbol: env.renames[n.fVar.symbol.name])
    spliceRewrite(n.fStart, env)
    spliceRewrite(n.fEnd, env)
    spliceRewrite(n.fStep, env)
    spliceRewrite(n.fBody, env)
  of gpuCall:
    # the callee ident stays, nested device calls splice in inlineStmts
    for i, arg in n.cArgs.mpairs:
      spliceRewrite(arg, env)
  else:
    for ch in mitems(n):
      spliceRewrite(ch, env)

proc blockStatements(n: GpuAst): seq[GpuAst] =
  ## Statement list of a branch body, a single statement lifts into a list
  if n.kind == gpuBlock:
    n.statements
  else:
    @[n]

proc toBlock(stmts: seq[GpuAst]): GpuAst =
  GpuAst(kind: gpuBlock, statements: stmts)

proc hoistExpr(ctx: var GpuContext, st: var KernelSplicer, pre: var seq[GpuAst], n: GpuAst): GpuAst
proc inlineStmts(ctx: var GpuContext, st: var KernelSplicer, stmts: seq[GpuAst]): seq[GpuAst]

proc inlineCall(ctx: var GpuContext, st: var KernelSplicer, call: GpuAst, retBind: GpuAst): seq[GpuAst] =
  ## Splices one device function body at one call site.
  ## - `retBind` receives the callee's implicit `result` bindings, nil for void callees
  ## Returns: the statements replacing the call, the parameter bindings first,
  ## nested device calls already spliced
  doAssert call.kind == gpuCall
  let callee = ctx.allFnTab[call.cName]
  doAssert callee.kind == gpuProc, "no device function body for " & call.cName.symbol.name
  doAssert not callee.pBody.isNil and callee.pBody.kind == gpuBlock
  let name = call.cName.symbol.name
  if call.cName.symbol.iSym in st.inlineStack:
    error "HLSL: recursive device function `" & name & "` cannot inline, inlining would not terminate"
  if call.cArgs.len != callee.pParams.len:
    error "HLSL: device function `" & name & "` takes " & $callee.pParams.len &
      " parameters, the call passes " & $call.cArgs.len
  let nonVoid = callee.pRetType.kind != gtVoid
  if nonVoid and retBind.isNil:
    error "HLSL: non-void device function `" & name & "` needs a result target"
  st.inlineStack.add call.cName.symbol.iSym

  # positional parameter binding: pointer parameters bind their uses to the argument's address target,
  # by-value parameters bind once to a fresh copy, by-reference parameters substitute their argument at each use
  var env = SpliceEnv(calleeName: name, usedNames: st.usedNames)
  if nonVoid:
    env.retName = "result"
    env.retBind = retBind
  var pre: seq[GpuAst]
  for i, prm in callee.pParams:
    let pname = prm.ident.symbol.name
    let arg = call.cArgs[i]
    if prm.typ.kind == gtPtr:
      case arg.kind
      of gpuAddr:
        env.ptrBind[pname] = arg.aOf
      of gpuIdent:
        if arg.symbol.typ.isNil or arg.symbol.typ.kind != gtPtr:
          error "HLSL: pointer parameter `" & pname & "` of device function `" &
            name & "` needs a pointer or address argument"
        env.ptrBind[pname] = arg
        env.ptrValues.incl pname
      else:
        error "HLSL: pointer parameter `" & pname & "` of device function `" &
          name & "` needs a pointer or address argument, got " & $arg.kind
    elif prm.passByRef:
      # a hidden const reference: the argument substitutes at each use,
      # a materialized argument unwraps to the value it materializes
      env.subst[pname] = if arg.kind == gpuMaterialize: arg.mExpr else: arg
    else:
      # by value: the argument seeds a fresh mutable copy, the callee's writes hit the copy
      let initExpr = if arg.kind == gpuMaterialize: arg.mExpr else: arg
      let copyName = st.freshName(pname)
      let copySym = newSymbol(copyName, iSym = copyName, typ = prm.typ, symKind = gsLocal)
      pre.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: copySym),
                     vType: prm.typ, vInit: initExpr.clone(), vMutable: true,
                     addressSpace: asRMEM)
      env.subst[pname] = GpuAst(kind: gpuIdent, symbol: copySym)

  var body = callee.pBody.clone()
  # a return anywhere but the trailing position would exit mid-splice,
  # the statements after it would run unconditionally
  for i, stmt in body.statements:
    if (stmt.kind == gpuReturn and i != body.statements.high) or
        (stmt.kind != gpuReturn and stmt.containsReturnStmt()):
      error "HLSL: early return in device function `" & name &
        "` cannot inline, only a trailing return splices"

  var declNames: seq[string]
  var declTypes = initTable[string, GpuType]()
  body.collectLocalDecls(declNames, declTypes)
  let retName = env.retName
  for local in declNames:
    if nonVoid and local == retName:
      continue
    let newName = st.freshName(local)
    env.renames[local] = newSymbol(newName, iSym = newName,
                                   typ = declTypes.getOrDefault(local), symKind = gsLocal)
  body.spliceRewrite(env)
  discard st.inlineStack.pop()
  # the dropped result decl and the dropped trailing return leave discards
  result = pre & ctx.inlineStmts(st, body.statements).filterIt(it.kind != gpuDiscard)

proc hoistExpr(ctx: var GpuContext, st: var KernelSplicer, pre: var seq[GpuAst], n: GpuAst): GpuAst =
  ## Lowers every device call inside an expression subtree: children first,
  ## then the node's own call splices onto a fresh temporary declared in `pre`.
  ## Returns: the expression to use in place of `n`, device calls become temporaries
  case n.kind
  of gpuCall:
    if not ctx.isSpliceableCall(n):
      for i, arg in n.cArgs.mpairs:
        n.cArgs[i] = ctx.hoistExpr(st, pre, arg)
      return n
    let callee = ctx.allFnTab[n.cName]
    if callee.pRetType.kind == gtVoid:
      error "HLSL: void device function `" & n.cName.symbol.name &
        "` in expression position cannot inline, a void call has no value"
    let tmp = st.freshName("call")
    let tmpSym = newSymbol(tmp, iSym = tmp, typ = callee.pRetType, symKind = gsLocal)
    pre.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: tmpSym),
                   vType: callee.pRetType, vInit: GpuAst(kind: gpuDiscard),
                   vMutable: true, addressSpace: asRMEM)
    pre.add ctx.inlineCall(st, n, GpuAst(kind: gpuIdent, symbol: tmpSym))
    result = GpuAst(kind: gpuIdent, symbol: tmpSym)
  of gpuBinOp:
    # `&&`/`||` evaluate their right operand conditionally, a call there
    # would splice unconditionally ahead of the short-circuit guard
    if not n.bOp.symbol.isNil and n.bOp.symbol.name in ["&&", "||"] and
        ctx.containsSpliceableCall(n.bRight):
      error "HLSL: device function call in a short-circuit right operand cannot inline, " &
        "the right operand evaluates conditionally"
    result = n
    for ch in mitems(result):
      ch = ctx.hoistExpr(st, pre, ch)
  of gpuTernary:
    # the condition evaluates once, its device calls hoist legally, the branches do not
    if ctx.containsSpliceableCall(n.tThen) or ctx.containsSpliceableCall(n.tElse):
      error "HLSL: device function call in a ternary branch cannot inline, " &
        "the branch evaluates conditionally"
    result = n
    for ch in mitems(result):
      ch = ctx.hoistExpr(st, pre, ch)
  of gpuAddr:
    if ctx.containsSpliceableCall(n.aOf):
      error "HLSL: address-of a device function call cannot inline"
    result = n
  else:
    result = n
    for ch in mitems(result):
      ch = ctx.hoistExpr(st, pre, ch)

proc inlineStmts(ctx: var GpuContext, st: var KernelSplicer, stmts: seq[GpuAst]): seq[GpuAst] =
  ## Splices statement-position device calls, hoists expression-position calls onto temps, nested callees splice transitively
  for s in stmts:
    case s.kind
    of gpuCall:
      if ctx.isSpliceableCall(s):
        let callee = ctx.allFnTab[s.cName]
        if callee.pRetType.kind == gtVoid:
          result.add ctx.inlineCall(st, s, nil)
        else:
          # a discarded non-void call still evaluates: splice onto a throwaway temp
          let tmp = st.freshName("call")
          let tmpSym = newSymbol(tmp, iSym = tmp, typ = callee.pRetType, symKind = gsLocal)
          result.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: tmpSym),
                            vType: callee.pRetType, vInit: GpuAst(kind: gpuDiscard),
                            vMutable: true, addressSpace: asRMEM)
          result.add ctx.inlineCall(st, s, GpuAst(kind: gpuIdent, symbol: tmpSym))
      else:
        result.add s
    of gpuVar:
      if s.vInit.kind == gpuCall and ctx.isSpliceableCall(s.vInit):
        # the initializer's callee splices straight onto the declared local
        let call = s.vInit
        s.vInit = GpuAst(kind: gpuDiscard)
        result.add s
        result.add ctx.inlineCall(st, call, s.vName)
      else:
        var pre: seq[GpuAst]
        s.vInit = ctx.hoistExpr(st, pre, s.vInit)
        result.add pre
        result.add s
    of gpuAssign:
      if s.aOp == "=" and s.aRight.kind == gpuCall and ctx.isSpliceableCall(s.aRight):
        # the callee splices straight onto the assignment target
        result.add ctx.inlineCall(st, s.aRight, s.aLeft)
      else:
        var pre: seq[GpuAst]
        s.aRight = ctx.hoistExpr(st, pre, s.aRight)
        result.add pre
        result.add s
    of gpuIf:
      var pre: seq[GpuAst]
      s.ifCond = ctx.hoistExpr(st, pre, s.ifCond)
      for el in s.ifElifs.mitems:
        # an elif condition evaluates only when reached, its device calls cannot hoist before the if
        if ctx.containsSpliceableCall(el.cond):
          error "HLSL: device function call in an elif condition cannot inline, " &
            "the condition evaluates only when reached"
        el.cond = ctx.hoistExpr(st, pre, el.cond)
      result.add pre
      s.ifThen = toBlock(ctx.inlineStmts(st, blockStatements(s.ifThen)))
      for el in s.ifElifs.mitems:
        el.body = toBlock(ctx.inlineStmts(st, blockStatements(el.body)))
      if s.ifElse.kind != gpuDiscard:
        s.ifElse = toBlock(ctx.inlineStmts(st, blockStatements(s.ifElse)))
      result.add s
    of gpuFor:
      if ctx.containsSpliceableCall(s.fStart) or ctx.containsSpliceableCall(s.fEnd) or
          ctx.containsSpliceableCall(s.fStep):
        error "HLSL: device function call in a for-loop header cannot inline, " &
          "the call would hoist out of the loop scope"
      s.fBody = toBlock(ctx.inlineStmts(st, blockStatements(s.fBody)))
      result.add s
    of gpuWhile:
      if ctx.containsSpliceableCall(s.wCond):
        error "HLSL: device function call in a while condition cannot inline, " &
          "the condition re-evaluates per iteration"
      s.wBody = toBlock(ctx.inlineStmts(st, blockStatements(s.wBody)))
      result.add s
    of gpuBlock:
      s.statements = ctx.inlineStmts(st, s.statements)
      result.add s
    of gpuEmit:
      var pre: seq[GpuAst]
      for i, part in s.parts.mpairs:
        if part.kind == peExpr:
          s.parts[i].expr = ctx.hoistExpr(st, pre, part.expr)
      result.add pre
      result.add s
    of gpuConstexpr:
      var pre: seq[GpuAst]
      s.cValue = ctx.hoistExpr(st, pre, s.cValue)
      result.add pre
      result.add s
    of gpuComment, gpuReturn, gpuDiscard, gpuInlineAsm:
      result.add s
    else:
      var pre: seq[GpuAst]
      let hoisted = ctx.hoistExpr(st, pre, s)
      result.add pre
      result.add hoisted

proc collectCallTargetISyms(n: GpuAst, targets: var HashSet[string]) =
  ## iSyms of every callee named in the statement tree `n`
  if n.isNil:
    return
  if n.kind == gpuCall and not n.cName.isNil and not n.cName.symbol.isNil:
    targets.incl n.cName.symbol.iSym
  for ch in n:
    collectCallTargetISyms(ch, targets)

proc inlineKernelCallees(ctx: var GpuContext) =
  ## Splices every non-intrinsic callee reachable from each kernel body into the kernel body, then drops the dead fnTab entries.
  ##
  ## - materialized numeric builtins keep their fnTab entry, the call prints the backend intrinsic
  ## - a fn no kernel reaches transitively drops, a fn only an unreachable fn calls is unreachable too
  for fnKey in toSeq(ctx.fnTab.keys):
    let fn = ctx.fnTab[fnKey]
    if fn.kind != gpuProc or not fn.isGlobal():
      continue
    var st = KernelSplicer()
    fn.pBody.collectIdentNames(st.usedNames)
    for p in fn.pParams:
      st.usedNames.incl p.ident.symbol.name
    fn.pBody.statements = ctx.inlineStmts(st, fn.pBody.statements)

  # the sweep iterates until the fnTab stops shrinking: a fn whose only callers
  # are themselves unreachable drops in a later round
  var changed = true
  while changed:
    changed = false
    var targets: HashSet[string]
    for fn in ctx.fnTab.values:
      fn.pBody.collectCallTargetISyms(targets)
    for fnKey in toSeq(ctx.fnTab.keys):
      let fn = ctx.fnTab[fnKey]
      if fn.kind != gpuProc or fn.isGlobal():
        continue
      if numericBuiltinDisplayName(fn.pName.ident()).len > 0:
        continue
      if fn.pName.symbol.iSym notin targets:
        ctx.fnTab.del(fnKey)
        changed = true
