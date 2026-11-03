# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Compile-time kernel-pair artifact of `bwd_diff`.
## The artifact is the printed sources of the pair plus the signature metadata a runtime backend needs, nothing else.
## No Nim AST is stored, expansion is a pure function of the kernel symbol.
##
## Backward signature rule, authoritative across the whole pair ABI:
##
## | Block         | Content                                                                    |
## | ------------- | -------------------------------------------------------------------------- |
## | backward      | `<d_-block>, <forward signature verbatim>`                                 |
## | d_-block      | one gradient slot per forward parameter with a gradient, forward order     |
## | buffer grads  | same shape as the primal buffer (`d_<name>`)                               |
## | scalar grads  | 1-element device buffers (`kind = buffer(1)`), matched by name, never type |
## | block 2       | the forward signature verbatim, forward outputs at their forward position  |
## | `save:` order | the forward output order                                                   |
## | workspaces    | reduce-partials live only in the backward kernel's IR signature            |
##
## A declared backward (`bwd_diff(fwd, bwd, save = [...])`) skips the derivation.
## Expansion validates it against this signature rule: gradient slots appear
## in forward parameter order, block 2 is the forward signature verbatim, and the kernel
## is named `<fwd>_bwd`.
##
## | Section    | Symbols                                                          |
## | ---------- | ---------------------------------------------------------------- |
## | Artifact   | `BwdDiffKernels`                                                 |
## | Metadata   | `BwdParamMeta`, `BwdParamKind`, `BwdGradKind`, `BwdExtentSource` |
## | Derivation | `deriveBwdParams` (declared pairs)                               |
## | Pipeline   | `codegenBwdDiffKernels`, `codegenBwdDiffKernelsDeclared`         |

import std/[macros, sets, strutils, tables]

import ../crucible_codegen
import ./autodiff_types
import ./backends/backends_api
import ./frontend/nim_to_crucible
import ./ir/cirv2_lang
import ./ir/cirv2_pretty
import ./passes/pass_helpers
import ./passes/pass_registry
import ./passes/passes_autodiff
import ./passes/passes_lowering_glsl
import ./passes/passes_lowering_msl
import ./passes/passes_normalizations
import ./passes/passes_validations

export crucible_codegen.CodegenKind
export autodiff_types.BwdParamKind, autodiff_types.BwdGradKind, autodiff_types.BwdExtentSource
export autodiff_types.BwdParamMeta, autodiff_types.ScalarWorkspace, autodiff_types.AdKernelsMetas

# ═══════════════════════════════════════════════════════════════════════
# ▸ Types
# ═══════════════════════════════════════════════════════════════════════

type AdWorkgroup* = object
  ## Launch geometry of the pair's kernels: the forward's declared workgroup,
  ## its writes proven thread-local by the derivation.
  ##
  ## A named object, not a tuple: the pair is a `static` parameter value,
  ## and Nim instantiates one generic handle per static value.
  ## Structurally equal tuples are distinct types to that machinery.
  x*, y*, z*: int

type ReduceSource* = object
  ## One generated reduce kernel of the pair: its name and standalone source.
  name*: string
  source*: string

type BwdDiffKernels* = object
  ## Const artifact of one `bwd_diff` expansion:
  ##
  ## - two printed sources
  ## - the reduce kernels of the scalar gradients
  ## - the parameter metadata
  ## - the launch geometry the pair's kernels bake and the launches pass
  kernelName*: string                        ## Forward kernel's name
  workgroup*: AdWorkgroup                    ## The kernels' baked workgroup, one launch geometry for every AD launch
  fwdSource*: string                         ## Forward kernel source, standalone translation unit
  bwdSource*: string                         ## Backward kernel source, `<kernel>_bwd`
  reduceSources*: seq[ReduceSource]          ## One reduce kernel per scalar gradient
  params*: seq[BwdParamMeta]                 ## Forward parameters in declaration order
  scalarWorkspaces*: seq[ScalarWorkspace]    ## Reduce-partials scratch per scalar gradient, forward order
  bwdKernelParams*: seq[string]              ## Printed backward kernel's parameter names, post-lowering, in binding order

# ═══════════════════════════════════════════════════════════════════════
# ▸ IR helpers
# ═══════════════════════════════════════════════════════════════════════

proc firstLoopBound(ss: seq[CirStmt]): CirExpr =
  ## Trip-count expression of the first desugared-for loop in statement order,
  ## nil when there is none. The differentiable subset refuses nested loops,
  ## so the derived path has at most one.
  var i = 0
  while i < ss.len:
    if i + 2 < ss.len and ss[i].kind == sVar and ss[i].init == nil and
        ss[i + 1].kind == sAsgn and ss[i + 1].lhs.kind == eSym and
        ss[i + 1].lhs.name == ss[i].name and ss[i + 2].kind == sWhile:
      return ss[i + 2].whileCond.b
    case ss[i].kind
    of sIte:
      let thenBound = firstLoopBound(ss[i].thenBody)
      if not thenBound.isNil:
        return thenBound
      let elseBound = firstLoopBound(ss[i].elseBody)
      if not elseBound.isNil:
        return elseBound
    of sWhile:
      let bodyBound = firstLoopBound(ss[i].whileBody)
      if not bodyBound.isNil:
        return bodyBound
    else:
      discard
    i.inc
  return nil

proc bufferExtent(p: CirParam, fwd: CirProc, paramNames: HashSet[string]): BwdParamMeta =
  ## Length source of one buffer parameter by order of authority: the parameter's own type, then the index-loop bound, then the wrapper.
  result = BwdParamMeta(name: p.name, extent: xeWrapperField)
  if p.typ.kind == ctPtr and p.typ.pElem != nil and p.typ.pElem.kind == ctArray:
    result.extent = xeStaticShape
    result.staticLen = p.typ.pElem.aN
    return
  let bound = firstLoopBound(fwd.body)
  if not bound.isNil and bound.kind == eSym and bound.name in paramNames:
    result.extent = xeLenParam
    result.extentParam = bound.name

# ═══════════════════════════════════════════════════════════════════════
# ▸ Thread-local write analysis
# ═══════════════════════════════════════════════════════════════════════

type ThreadTermEnv = object
  ## Local definitions of one kernel body for the thread-term analysis.
  ## A local with more than one defining event is ambiguous and never resolves.
  defs: Table[string, CirExpr]
  multi: HashSet[string]

proc collectThreadTermEnv(fn: CirProc): ThreadTermEnv =
  ## Collects the local definitions of one kernel body from local initializers
  ## and ordinary local assignments, nested statement lists included.
  var env: ThreadTermEnv
  let onStmt = proc(s: CirStmt) =
    case s.kind
    of sVar, sConst:
      if env.defs.hasKey(s.name):
        env.multi.incl s.name
      else:
        env.defs[s.name] = s.init
    of sAsgn:
      if s.lhs.kind == eSym:
        if env.defs.hasKey(s.lhs.name):
          env.multi.incl s.lhs.name
        else:
          env.defs[s.lhs.name] = s.rhs
    else:
      discard
  fn.body.walkStmts(onStmt, nil)
  result = env

type ThreadTerm = tuple[coeff, constPart: int, coeffOpaque, constOpaque: bool]

proc affineThreadTerm(e: CirExpr, env: ThreadTermEnv, depth: int, ok: var bool): ThreadTerm =
  ## Affine decomposition of an index expression in the thread-identity term:
  ## `coeff * t + constPart`. `ok` turns false when the expression is not
  ## provably affine with a literal thread coefficient.
  ##
  ## - a coordinate-builtin read is the thread term, coefficient 1
  ## - a local resolves to its defining expression, one defining event only
  ## - any other identifier is a thread-invariant constant of unknown value,
  ##   an opaque constant never proves a nonzero literal coefficient
  ## - integer `+`, `-` and `*` combine terms, `*` needs one constant factor,
  ##   every other node kind refuses
  if not ok or depth > 32:
    ok = false
    return
  case e.kind
  of eIntLit:
    result.constPart = e.value.int
  of eSym:
    if e.builtin and e.coordBuiltin != cbkNone:
      result.coeff = 1
    elif env.defs.hasKey(e.name) and e.name notin env.multi:
      let def = env.defs[e.name]
      if def.isNil:
        ok = false
      else:
        result = affineThreadTerm(def, env, depth + 1, ok)
    else:
      result.constOpaque = true
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.builtin and
        e.dotBase.coordBuiltin != cbkNone:
      result.coeff = 1
    else:
      ok = false
  of eAdd, eSub, eMul:
    let l = affineThreadTerm(e.left, env, depth, ok)
    let r = affineThreadTerm(e.right, env, depth, ok)
    if not ok:
      return
    case e.kind
    of eAdd:
      result.coeff = l.coeff + r.coeff
      result.coeffOpaque = l.coeffOpaque or r.coeffOpaque
      result.constPart = l.constPart + r.constPart
      result.constOpaque = l.constOpaque or r.constOpaque
    of eSub:
      result.coeff = l.coeff - r.coeff
      result.coeffOpaque = l.coeffOpaque or r.coeffOpaque
      result.constPart = l.constPart - r.constPart
      result.constOpaque = l.constOpaque or r.constOpaque
    of eMul:
      let lConst = l.coeff == 0 and not l.coeffOpaque
      let rConst = r.coeff == 0 and not r.coeffOpaque
      if lConst and rConst:
        result.constPart = l.constPart * r.constPart
        result.constOpaque = l.constOpaque or r.constOpaque
      elif lConst and r.coeff != 0 and not r.coeffOpaque:
        result.coeff = r.coeff * l.constPart
        result.coeffOpaque = l.coeffOpaque
        result.constPart = r.constPart * l.constPart
        result.constOpaque = l.constOpaque or r.constOpaque
      elif rConst and l.coeff != 0 and not l.coeffOpaque:
        result.coeff = l.coeff * r.constPart
        result.coeffOpaque = r.constOpaque
        result.constPart = l.constPart * r.constPart
        result.constOpaque = r.constOpaque or l.constOpaque
      else:
        ok = false
    else:
      discard
  of eConv:
    result = affineThreadTerm(e.castOperand, env, depth, ok)
  else:
    ok = false

proc checkThreadLocalWrites(fn: CirProc, role: string, wg: array[3, int]) =
  ## Refuses a kernel of the pair whose body writes a buffer slot at an index
  ## the affine analysis cannot prove thread-local.
  ##
  ## - a multi-thread workgroup runs such a write from every thread,
  ##   the threadgroup slots race
  ## - atomic accumulation statements are race-safe by construction, they skip
  let termEnv = collectThreadTermEnv(fn)
  let onStmt = proc(s: CirStmt) =
    if s.kind == sAsgn and s.lhs.kind in {ePat, eAt}:
      let root = getBufferRoot(s.lhs)
      if root != nil and root.kind == eSym:
        var ok = true
        let idx = if s.lhs.kind == ePat: s.lhs.patIndex else: s.lhs.atIndex
        let term = affineThreadTerm(idx, termEnv, 0, ok)
        if not ok or term.coeffOpaque or term.coeff <= 0:
          error "bwd_diff: kernel '" & fn.name & "' at workgroup (" & $wg[0] & ", " & $wg[1] & ", " & $wg[2] & "): " & role & " writes '" & renderExpr(s.lhs) & "', the write index does not separate the threads, the threadgroup races on it"
  fn.body.walkStmts(onStmt, nil)

proc enforceBwdWorkgroup(fwd, bwd: CirProc, reduces: seq[CirProc]) =
  ## Launch geometry of the pair: the forward's declared workgroup,
  ## unset axes normalized to 1.
  ##
  ## - an unset or one-thread annotation keeps the v1 shape:
  ##   every kernel of the pair bakes a 1x1x1 workgroup and every AD launch
  ##   passes that geometry explicitly
  ## - a multi-thread annotation requires every buffer-slot write of the forward
  ##   and of the backward to carry an index the affine analysis proves thread-local
  ## - a write the analysis cannot prove local refuses at expansion,
  ##   the message names the statement
  ## - the reduce kernels always stay single-thread:
  ##   their sequential partials fold must never multiply across a workgroup
  var wg = fwd.workgroup
  for d in 0 ..< 3:
    if wg[d] == 0:
      wg[d] = 1
  if wg == [1, 1, 1]:
    for fn in @[fwd, bwd] & reduces:
      fn.workgroup = [1, 1, 1]
    return
  checkThreadLocalWrites(fwd, "the forward", wg)
  checkThreadLocalWrites(bwd, "the backward", wg)
  fwd.workgroup = wg
  bwd.workgroup = wg
  for rk in reduces:
    rk.workgroup = [1, 1, 1]

# ═══════════════════════════════════════════════════════════════════════
# ▸ Declared backward
# ═══════════════════════════════════════════════════════════════════════
#
# Declared path: `bwd_diff(fwd, bwd, save = [...])` skips the derivation.
# A hand-written backward kernel serves as the backward, and the backward
# signature rule still governs its signature, derived from the forward symbol alone.
# A mismatch refuses at expansion.

proc collectWrittenParams(fn: CirProc, paramNames: HashSet[string], written: var HashSet[string]) =
  ## Buffer parameters the kernel body writes: assignment targets and atomic
  ## statement targets whose root is one of `paramNames`.
  var collected = written
  let onStmt = proc(s: CirStmt) =
    case s.kind
    of sAsgn:
      if s.lhs.kind in {ePat, eAt}:
        let root = getBufferRoot(s.lhs)
        if root != nil and root.name in paramNames:
          collected.incl root.name
    of sCall:
      let callee = s.callCallee
      if callee.kind == eSym and callee.builtin and
          callee.atomicsBuiltin != cbkNone and
          s.callArgs.len == 2 and s.callArgs[0].kind == eAddr:
        let root = getBufferRoot(s.callArgs[0].addrOperand)
        if root != nil and root.name in paramNames:
          collected.incl root.name
    else:
      discard
  fn.body.walkStmts(onStmt, nil)
  written = collected

proc deriveBwdParams(fwd, bwd: CirProc): seq[BwdParamMeta] =
  ## Derives the backward-signature metadata for a DECLARED pair from the two
  ## post-pass procs. The derived path does not run this proc: its metadata is
  ## the derivation's own record (`m.adMetas`), derived once from the walk that built
  ## the backward kernel.
  ##
  ## Expected input:
  ## - `fwd`: the forward kernel as the pipeline produced it
  ## - `bwd`: the hand-written backward kernel, `<fwd>_bwd`
  ##
  ## A float scalar gets `gkBuffer1` when the backward's parameter list contains
  ## a `d_<name>` slot. The hand-written kernel writes the 1-element buffer itself,
  ## no reduce kernel exists.
  ##
  ## Output: one `BwdParamMeta` per forward parameter, declaration order.
  var paramNames: HashSet[string]
  for p in fwd.params:
    paramNames.incl p.name
  # Gradient slots match by parameter name, never by symbol id:
  # symbol ids live in a per-proc namespace, so a declared backward is a proc
  # of its own with different ids across the two procs even under equal names.
  # Validation already checked the two name lists positionally.
  var bwdParamNames: HashSet[string]
  for p in bwd.params:
    bwdParamNames.incl p.name
  var written: HashSet[string]
  fwd.collectWrittenParams(paramNames, written)

  for p in fwd.params:
    let name = p.name
    if getBufferElemType(p) != nil:
      var meta = p.bufferExtent(fwd, paramNames)
      if name in written:
        meta.kind = pkResidual
      else:
        meta.kind = pkBuffer
      if ("d_" & name) in bwdParamNames:
        meta.gradKind = gkBuffer
        meta.gradName = "d_" & name
      result.add meta
    elif isDiffFloat(p.typ):
      var meta = BwdParamMeta(name: name, kind: pkScalar)
      if ("d_" & name) in bwdParamNames:
        meta.gradKind = gkBuffer1
        meta.gradName = "d_" & name
      result.add meta
    else:
      result.add BwdParamMeta(name: name, kind: pkScalar)

proc mirrorGradientNames(fwd: CirProc): seq[string] =
  ## d_-block of the backward signature, derived from the forward symbol alone.
  ##
  ## Expected input:
  ## - `fwd`: the forward kernel as the pipeline produced it
  ##
  ## Output: one `d_<name>` slot per forward parameter that the forward body references,
  ## buffers and float scalars alike, in forward order.
  ##
  ## The derivation seeds its own d_-block from the same buffer walk,
  ## plus the scalar gradients it actually demands, so the two lists agree for kernels
  ## in the differentiable subset.
  ##
  ## One narrow case may differ: a float scalar referenced only in a zero-gradient
  ## position such as a comparison has no derived slot but does have a declared slot.
  var paramNames: HashSet[string]
  for p in fwd.params:
    paramNames.incl p.name
  var referenced: HashSet[string]
  let onExpr = proc(e: CirExpr) =
    if e.kind == eSym and not e.builtin and e.name in paramNames:
      referenced.incl e.name
  fwd.body.walkStmts(nil, onExpr)
  for p in fwd.params:
    if (getBufferElemType(p) != nil or isDiffFloat(p.typ)) and p.name in referenced:
      result.add "d_" & p.name

proc declaredMirrorSignature(fwd: CirProc): seq[string] =
  ## Full signature a declared backward must reproduce exactly:
  ## the d_-block of `mirrorGradientNames`, then the forward signature verbatim.
  result = fwd.mirrorGradientNames()
  for p in fwd.params:
    result.add p.name

proc validateDeclaredBwd(fwd, bwd: CirProc) =
  ## Refuses a declared backward that deviates from its forward's signature.
  ## The contract, checked in order:
  ##
  ## - the backward kernel is named `<fwd>_bwd`, and the pair's backward launches under that name
  ## - its parameters equal `declaredMirrorSignature(fwd)` positionally,
  ##   the refusal names the first differing position and the rule it breaks
  let fwdName = fwd.name
  let bwdName = bwd.name
  if bwdName != fwdName & "_bwd":
    error "bwd_diff: the declared backward '" & bwdName & "' must be named '" & fwdName & "_bwd', the pair's launch name"
  let mirror = fwd.declaredMirrorSignature()
  var declared: seq[string]
  for p in bwd.params:
    declared.add p.name
  if declared == mirror:
    return
  let dBlock = fwd.mirrorGradientNames().len
  var i = 0
  while i < declared.len and i < mirror.len and declared[i] == mirror[i]:
    i.inc
  let rule = if i < dBlock:
    "gradients sit in forward-parameter order"
  else:
    "block 2 is the forward signature verbatim"
  if i < declared.len and i < mirror.len:
    error "bwd_diff: the declared backward '" & bwdName & "' does not mirror '" & fwdName & "': parameter " & $(i + 1) & " is '" & declared[i] & "', the mirror rule derives '" & mirror[i] & "' (" & rule & ")"
  elif i < declared.len:
    error "bwd_diff: the declared backward '" & bwdName & "' does not mirror '" & fwdName & "': parameter " & $(i + 1) & " is '" & declared[i] & "', beyond the mirror (" & rule & ")"
  else:
    error "bwd_diff: the declared backward '" & bwdName & "' does not mirror '" & fwdName & "': it stops at parameter " & $declared.len & ", the mirror rule derives '" & mirror[i] & "' (" & rule & ")"

proc validateSaveNames(fwdName: string, metas: seq[BwdParamMeta], saveNames: seq[string]) =
  ## Refuses `save = [...]` names that are not forward outputs of the pair's
  ## forward kernel: a save names a buffer the forward kernel writes.
  for s in saveNames:
    var known = false
    for m in metas:
      if m.name == s:
        known = true
        if m.kind != pkResidual:
          error "bwd_diff: save name '" & s & "' is not a forward output of '" & fwdName & "', the forward kernel does not write it"
    if not known:
      error "bwd_diff: save name '" & s & "' is not a parameter of '" & fwdName & "'"

# ═══════════════════════════════════════════════════════════════════════
# ▸ Pipelines
# ═══════════════════════════════════════════════════════════════════════

proc refuseExpansion(what: string) =
  ## Surfaces a pipeline refusal at the `bwd_diff` expansion: messages that already
  ## include the `bwd_diff: ` prefix pass through unchanged, and every other
  ## refusal gains the prefix.
  if "bwd_diff: " in what:
    error what
  else:
    error "bwd_diff: " & what

proc printSingleton(m: CirModule, fn: CirProc, B: static CodegenKind): string =
  ## Prints one proc as a standalone translation unit of backend `B`.
  ##
  ## A singleton module over `fn` runs the backend lowering passes and the validation
  ## stage through its own registry, then prints. Every kernel owns its type
  ## declarations and push-constant block, so the multi-kernel shared-push-constant
  ## validation does not apply.
  let single = CirModule(name: fn.name, types: m.types, procs: @[fn])
  var reg = PassRegistry.new()
  when B == cgMetal:
    registerMetalLoweringPasses(reg)
  else:
    registerGlslLoweringPasses(reg)
  registerValidationPostPasses(reg)
  let err = runPasses(single, reg)
  if err.len > 0:
    refuseExpansion(err)
  result = codegen(single, B)

proc printPair(m: CirModule, B: static CodegenKind, fwd, bwd: CirProc,
               reduces: seq[CirProc]): tuple[fwdSource: string, bwdSource: string,
                                             reduceSources: seq[ReduceSource],
                                             bwdKernelParams: seq[string]] =
  ## Stage 2 of the pair pipelines: per-kernel standalone printing over backend lowering.
  ##
  ## The backward's parameters are recorded after its lowering run in printed
  ## parameter order. Coordinate builtins are excluded: the launcher binds those implicitly,
  ## and they take no call-site slot.
  for rk in reduces:
    result.reduceSources.add ReduceSource(name: rk.name,
                                          source: m.printSingleton(rk, B))
  result.bwdSource = m.printSingleton(bwd, B)
  for p in bwd.params:
    if p.name notin CoordBuiltinNames:
      result.bwdKernelParams.add p.name
  result.fwdSource = m.printSingleton(fwd, B)

proc codegenBwdDiffKernels*(B: static CodegenKind, body: NimNode): BwdDiffKernels =
  ## Pipeline of one `bwd_diff` expansion for backend `B`:
  ##
  ##   Nim body → frontend → normalization → backward derivation → metadata
  ##           → backend lowering → per-kernel print
  ##
  ## - the typed kernel body goes through the frontend and the normalization
  ##   passes and the backward derivation
  ## - the signature metadata is the derivation's own record (`m.adMetas`)
  ## - each kernel prints standalone after the backend lowering
  ##
  ## Compile-time only, called from the `bwd_diff` macro.
  ##
  ## A `static` backend-kind macro parameter cannot host the pipeline:
  ##
  ## - a static parameter makes the macro generic
  ## - instantiation re-sem-checks the typed kernel body
  ## - statement pragmas like `{.emit: ... .}` refuse there
  ## - the differentiable subset must diagnose those pragmas itself
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    refuseExpansion(lerr)
  var reg = PassRegistry.new()
  registerNormalizationPasses(reg)
  registerAutodiffPasses(reg)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    refuseExpansion(perr)

  # Pair pickup: the user kernel first, then the backward and reduce kernels
  # that the derivation generated. Every generated kernel is recorded under
  # `m.adBwdKernels`, and the user kernel is the one kernel outside that record.
  var fwd: CirProc = nil
  for fn in m.procs:
    if not fn.isKernel or fn.name in m.adBwdKernels:
      continue
    if not fwd.isNil:
      error "bwd_diff: more than one {.global.} kernel to differentiate: '" & fwd.name & "' and '" & fn.name & "'"
    fwd = fn
  if fwd.isNil:
    error "bwd_diff: no {.global.} kernel to differentiate"
  var bwd: CirProc = nil
  var reduces: seq[CirProc]
  for fn in m.procs:
    if fn.name == fwd.name:
      continue
    if fn.name == fwd.name & "_bwd":
      bwd = fn
    elif fn.name.startsWith(fwd.name & "_bwd_d_") and
        fn.name.endsWith("_reduce"):
      reduces.add fn
    else:
      error "bwd_diff: unexpected proc '" & fn.name & "' in the pair pipeline"
  if bwd.isNil:
    error "bwd_diff: no backward kernel was generated for '" & fwd.name & "'"

  # Metadata comes from the derivation's own record, derived once from the walk
  # that built the backward kernel, and the pipeline runs no second body walk,
  # no name-suffix matching against the generated reduce kernels.
  let recorded = m.adMetas[fwd.name]

  enforceBwdWorkgroup(fwd, bwd, reduces)

  let printed = m.printPair(B, fwd, bwd, reduces)
  result = BwdDiffKernels(
    kernelName: fwd.name,
    workgroup: AdWorkgroup(x: fwd.workgroup[0], y: fwd.workgroup[1],
                           z: fwd.workgroup[2]),
    fwdSource: printed.fwdSource,
    bwdSource: printed.bwdSource,
    reduceSources: printed.reduceSources,
    params: recorded.params,
    scalarWorkspaces: recorded.workspaces,
    bwdKernelParams: printed.bwdKernelParams)

proc codegenBwdDiffKernelsDeclared*(B: static CodegenKind, fwdBody, bwdBody: NimNode, saveNames: seq[string]): BwdDiffKernels =
  ## Pipeline of one declared-backward `bwd_diff(fwd, bwd, save = [...])`
  ## expansion for backend `B`:
  ##
  ##   Nim bodies → frontend → normalization → mirror validation
  ##             → backend lowering → per-kernel print
  ##
  ## Contract:
  ##
  ## - No autodiff pass runs: the hand-written kernel is the pair's backward.
  ## - The pair's metadata still derives from the forward symbol.
  ## - A declared backward whose signature deviates from the rule refuses
  ##   at expansion with the `bwd_diff:` prefix, see `validateDeclaredBwd`.
  ## - Every save name must be a forward output of the forward kernel, checked
  ##   by `validateSaveNames`. Saved outputs stay alive between the forward
  ##   and the backward.
  ##
  ## Compile-time only, called from the `bwd_diff` macro.
  let (m, lerr) = lowerCirv2(nnkStmtList.newTree(fwdBody, bwdBody))
  if lerr.len > 0:
    refuseExpansion(lerr)
  var reg = PassRegistry.new()
  registerNormalizationPasses(reg)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    refuseExpansion(perr)

  # Pair ingestion runs in declaration order, forward kernel first.
  var globals: seq[CirProc]
  for fn in m.procs:
    if fn.isKernel:
      globals.add fn
  if globals.len != 2:
    error "bwd_diff: a declared pair takes one {.global.} forward kernel and one '<forward>_bwd' backward, got " & $globals.len & " kernels"
  let fwd = globals[0]
  let bwd = globals[1]

  # Signature validation runs before derivation: the rule is the only authority.
  validateDeclaredBwd(fwd, bwd)
  let metas = deriveBwdParams(fwd, bwd)
  validateSaveNames(fwd.name, metas, saveNames)

  enforceBwdWorkgroup(fwd, bwd, @[])

  let printed = m.printPair(B, fwd, bwd, @[])
  result = BwdDiffKernels(
    kernelName: fwd.name,
    workgroup: AdWorkgroup(x: fwd.workgroup[0], y: fwd.workgroup[1],
                           z: fwd.workgroup[2]),
    fwdSource: printed.fwdSource,
    bwdSource: printed.bwdSource,
    reduceSources: printed.reduceSources,
    params: metas,
    scalarWorkspaces: @[],
    bwdKernelParams: printed.bwdKernelParams)
