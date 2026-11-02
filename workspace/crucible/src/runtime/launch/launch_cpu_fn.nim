# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## `launchCpuFn`: the CPU launch surface, CPU-native end to end.
##
## A kernel is a Nim proc launched with whole values.
## Launch pipeline at compile time: overload resolution and the compile-time/runtime parameter split, then CIRv2 lowering and emission.
##
## Generated shape, no runtime compiler, no FFI, no context struct:
##
##   one task per contiguous workgroup range     # K = workgroupsPerTask
##   for wg in the task range:                   # x-fastest flat order
##     gx/gy/gz from wg
##     for bz/by/bx:                             # threads run serially inside the task
##       <kernel body, coordinates read as loop-variable expressions>
##
## - `--threads:on`: the kernel machinery (c_lang C printer) emits to a generated
##   .c file registered {.compile.}, the inline launch tail packs the data struct, spawns the loop task, syncAll waits
##   the tail translation unit duplicates the struct definitions, identical defs per unit are legal C
## - without threads: serial spliced Nim loops on the caller thread, the parallel variant must match that output bitwise
##
## Schedule: a `do:` block with `grid = ...` / `blk = ...`, both optional.
##
## - both given: explicit schedule
## - `workgroupsPerTask = K`: K adjacent workgroups (x-fastest flat order)
##   share one pool task, default 1, one workgroup per task
## - a coarse launch runs ceil(workgroups / K) tasks and clamps the last task range at the grid total
##   the serial launch has no pool tasks, it ignores the knob
## - heuristic starting point: K ~ workgroups / (4 * pool size), about four tasks per worker
##
## The derived launch (neither given):
##
## - coordinate-free kernels run on the (1,1,1) x (1,1,1) schedule
## - coordinate kernels without a schedule refuse at compile time
## - tile-layout schedule derivation is unimplemented
##
## Other:
##
## - runtime formals hoist into locals under their own names, evaluated
##   once per launch, the pool task receives a copy of the values
## - each launch splices into a fresh block scope
## - the same kernel at two sites emits two copies under per-site suffixes

import std/[macros, os, strutils, tables]

import workspace/crucible/src/codegen/cpu_compiler
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/engines/cpu_api
# ═══════════════════════════════════════════════════════════════════════
# ▸ Launch-site helpers
# ═══════════════════════════════════════════════════════════════════════

func isCompileTime(ftype: NimNode): bool =
  ## True for `static T` and `typedesc` formal types, the parameters baked
  ## into the generated kernel.
  ftype.kind == nnkStaticTy or
    (ftype.kind in {nnkIdent, nnkSym} and ftype.eqIdent("typedesc")) or
    (ftype.kind in {nnkCommand, nnkBracketExpr, nnkCall} and
     $ftype[0] in ["static", "typedesc"])

func formalCounts(candFormals: NimNode): tuple[total, required: int] =
  ## Parameter counts of one candidate overload, default parameters count as optional.
  for i in 1 ..< candFormals.len:
    let f = candFormals[i]
    result.total += f.len - 2
    if f[^1].kind == nnkEmpty:
      result.required += f.len - 2

func resolveKernelOverload(kernel: NimNode, argCount: int): NimNode =
  ## Picks the overload of `kernel` whose parameters accept `argCount`
  ## runtime-call arguments. Exactly one must match.
  if kernel.kind notin {nnkOpenSymChoice, nnkClosedSymChoice}:
    return kernel
  var match: NimNode = nil
  var count = 0
  for cand in kernel:
    let candImpl = cand.getImpl()
    if candImpl.kind != nnkProcDef:
      continue
    let (total, required) = formalCounts(candImpl[3])
    if required <= argCount and argCount <= total:
      match = cand
      count.inc
  doAssert count == 1,
    "kernel launch " & $kernel & ": " & $count & " overloads take " &
    $argCount & " arguments"
  match

# ═══════════════════════════════════════════════════════════════════════
# ▸ Kernel wrapper
# ═══════════════════════════════════════════════════════════════════════

proc isDeviceMemArg(t: NimNode): bool =
  t.kind == nnkBracketExpr and t[0].eqIdent("DeviceMem")

proc buildCpuKernel(kernel: NimNode, statics, values: NimNode, siteSuffix: string): NimNode =
  ## Kernel wrapper shape for the CPU pipeline:
  ##
  ##   proc <kernel>_cpuk<suffix>(<runtime formals>) {.global.} =
  ##     <kernel>(<compile-time args>, <runtime formals>)
  ##
  ## Compile-time parameters bake in, runtime parameters re-declare typed
  ## from their arguments.
  ## `{.global.}` marks the kernel for the CIRv2 frontend.
  ## CPU code never generates from the wrapper proc itself.
  let impl = kernel.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalNames: seq[NimNode]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalNames.add(f[j])
      formalTypes.add(f[^2])
  var kcall = ident($kernel).newCall()
  var params = nnkFormalParams.newTree(newEmptyNode())
  var si = 0
  var vi = 0
  for i in 0 ..< formalTypes.len:
    if si >= statics.len and vi >= values.len:
      break
    if isCompileTime(formalTypes[i]):
      kcall.add(statics[si])
      si.inc
    else:
      let v = values[vi]
      if v.getTypeInst().isDeviceMemArg():
        error("launchCpuFn " & $kernel & ": parameter " & $formalNames[i].repr &
          " takes DeviceMem, a GPU runtime concept, the cpu engine takes host values")
      let pname = ident($formalNames[i])
      params.add(nnkIdentDefs.newTree(pname, v.getTypeInst(), newEmptyNode()))
      kcall.add(pname)
      vi.inc
  doAssert vi > 0, "launchCpuFn " & $kernel & ": the kernel needs a runtime parameter"
  result = nnkProcDef.newTree(
    ident($kernel & "_cpuk" & siteSuffix),
    newEmptyNode(), newEmptyNode(),
    params,
    nnkPragma.newTree(ident"global"),
    newEmptyNode(),
    nnkStmtList.newTree(kcall))

# ═══════════════════════════════════════════════════════════════════════
# ▸ Launch
# ═══════════════════════════════════════════════════════════════════════

proc forLoop(varName: string, bound: NimNode, body: NimNode): NimNode {.compileTime.} =
  nnkForStmt.newTree(ident(varName),
    nnkInfix.newTree(ident"..<", newLit(0), bound), body)

macro cpuExecLower(engine: CpuEngine, wrapper: typed, gridN, blkN, coarsenN: untyped, siteSuffix: static string, args: varargs[typed]): untyped =
  ## Pipeline and launch: CIRv2 lowering of the typed wrapper, Nim emission,
  ## and the loop nest with the kernel body inlined.
  ##
  ## - the wrapper's formals are the kernel's runtime formals verbatim
  ## - runtime arguments hoist into locals under those names, one evaluation per launch
  ##   an argument that is the formal name itself skips the hoist,
  ##   the user variable stays in scope and the body reads it directly
  ## - schedule: both nil means derivation, coordinate kernels refuse there
  ## - with `--threads:on` each pool task covers workgroupsPerTask workgroups,
  ##   adjacent in the x-fastest flat order matching the serial decomposition
  ##   without threads the serial nested loops run, the executable spec
  let formals = wrapper[3]
  var runtimeNames: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      runtimeNames.add(f[j])
  doAssert args.len == runtimeNames.len,
    "launchCpuFn: " & $args.len & " runtime args for " &
    $runtimeNames.len & " runtime parameters"

  let m = lowerCpuKernel(wrapper)

  var gridNeff = gridN
  var blkNeff = blkN
  if gridN.kind == nnkNilLit and blkN.kind == nnkNilLit:
    gridNeff = nnkPar.newTree(newLit(1), newLit(1), newLit(1))
    blkNeff = nnkPar.newTree(newLit(1), newLit(1), newLit(1))
  elif gridN.kind == nnkNilLit:
    gridNeff = nnkPar.newTree(newLit(1), newLit(1))
  elif blkN.kind == nnkNilLit:
    blkNeff = nnkPar.newTree(newLit(1), newLit(1))
  var coarsenNeff = coarsenN
  if coarsenNeff.kind == nnkNilLit:
    coarsenNeff = newLit(1)
  let gridVar = ident"crucibleGrid"
  let blkVar = ident"crucibleBlk"
  let gridDecl = nnkVarSection.newTree(
    nnkIdentDefs.newTree(gridVar, newEmptyNode(), bindSym"dim3".newCall(gridNeff)))
  let blkDecl = nnkVarSection.newTree(
    nnkIdentDefs.newTree(blkVar, newEmptyNode(), bindSym"dim3".newCall(blkNeff)))

  proc gridDot(field: string): NimNode = nnkDotExpr.newTree(gridVar, ident(field))
  proc blkDot(field: string): NimNode = nnkDotExpr.newTree(blkVar, ident(field))

  var hoists = nnkStmtList.newTree()
  for i, nameNode in runtimeNames:
    let name = $nameNode
    let arg = args[i]
    if not (arg.kind == nnkIdent and $arg == name):
      hoists.add(nnkVarSection.newTree(
        nnkIdentDefs.newTree(ident(name), newEmptyNode(), arg)))

  # launch shape: workgroup ranges through the C runtime with --threads:on,
  # serial nested loops otherwise
  var launchBody = nnkStmtList.newTree(gridDecl, blkDecl, hoists)
  when compileOption("threads"):
    # workgroups per pool task. The C iter fn and the launch tail read it
    launchBody.add(nnkLetSection.newTree(
      nnkIdentDefs.newTree(ident"crucibleWgsPerTask", newEmptyNode(),
        nnkCall.newTree(bindSym"int32", coarsenNeff))))
    launchBody.add(nnkCall.newTree(bindSym"doAssert",
      nnkInfix.newTree(ident">=", ident"crucibleWgsPerTask", newLit(1'i32)),
      newLit("launchCpuFn: workgroupsPerTask must be >= 1")))
    # whole workgroup launch, C text against the threadpool_compute runtime.
    # c_lang prints every piece, the launch splits the text in two:
    # - file-scope unit: nominal struct definitions, the per-launch data struct, the callees, the iteration function, the task function
    # - launch tail: data struct fill, task spawn, syncAll, inline emit
    #   function definitions stay out of the enclosing function body:
    #   block-scope function definitions are a GNU extension, host C compilers reject them
    let gridReads = ["tpc_d_->crucibleGrid[0]", "tpc_d_->crucibleGrid[1]",
                     "tpc_d_->crucibleGrid[2]"]
    let blkReads = ["tpc_d_->crucibleBlk[0]", "tpc_d_->crucibleBlk[1]",
                    "tpc_d_->crucibleBlk[2]"]
    var vecC: array[CirCoordBuiltinKind, array[3, string]]
    vecC[cbkThreadgroupPositionInGrid] = ["crucibleGx", "crucibleGy", "crucibleGz"]
    vecC[cbkThreadPositionInThreadgroup] = ["crucibleBx", "crucibleBy", "crucibleBz"]
    vecC[cbkThreadPositionInGrid] = [
      "(crucibleGx * " & blkReads[0] & " + crucibleBx)",
      "(crucibleGy * " & blkReads[1] & " + crucibleBy)",
      "(crucibleGz * " & blkReads[2] & " + crucibleBz)"]
    vecC[cbkThreadsPerThreadgroup] = blkReads
    vecC[cbkThreadgroupsPerGrid] = gridReads
    let flatC = "(crucibleBx + crucibleBy * " & blkReads[0] &
                " + crucibleBz * " & blkReads[0] & " * " & blkReads[1] & ")"
    let coordsC = cpuCoordTableC(vecC, flatC)

    let runtimeParams = kernelRuntimeParams(m)
    let (typesAndData, funcsC, coordUseC) = emitCpuWorkgroupLaunchC(
        m, siteSuffix, coordsC, runtimeParams)
    if gridN.kind == nnkNilLit and blkN.kind == nnkNilLit and coordUseC:
      error("launchCpuFn " & $wrapper[0] & ": kernel reads coordinate builtins and the cpu backend cannot derive the schedule, pass grid/blk in the do block")
    let calleesC = emitCpuCalleesC(m, siteSuffix)

    # file-scope unit, written at macro time under the project and registered
    # {.compile.}. Nim hashes the file, so a changed kernel recompiles
    # while identical text reuses the object
    var unitC = typesAndData
    if calleesC.len > 0:
      unitC.add(calleesC & "\n")
    unitC.add(funcsC)
    let genDir = getProjectPath() / "crucibleGenerated"
    createDir(genDir)
    let genPath = genDir / ("crucibleLaunch_" & siteSuffix & ".c")
    writeFile(genPath, unitC)
    launchBody.add(parseStmt("{.compile: \"" & genPath & "\".}"))

    # launch tail: the struct definitions duplicate into the tail translation unit,
    # identical definitions per unit are legal C. The data struct receives
    # grid/blk from the dim3 locals of the launch and the runtime arguments,
    # Nim value objects are C-compatible structs, a struct argument copies
    # by pointer cast. Nim symbols go inside emit backticks, the codegen substitutes their C names,
    # immune to the generator's renaming
    var ctext = typesAndData
    ctext.add("    extern void crucibleTask" & siteSuffix & "(void*);\n")
    ctext.add("    tpc_worker_context* crucibleCtx = tpc_workerContext();\n")
    ctext.add("    struct crucibleData" & siteSuffix & " crucibleData;\n")
    for (c, ci) in [("x", 0), ("y", 1), ("z", 2)]:
      ctext.add("    crucibleData.crucibleGrid[" & $ci & "] = (int32_t)`crucibleGrid`." & c & ";\n")
      ctext.add("    crucibleData.crucibleBlk[" & $ci & "] = (int32_t)`crucibleBlk`." & c & ";\n")
    ctext.add("    crucibleData.crucibleWgsPerTask = (int32_t)`crucibleWgsPerTask`;\n")
    for p in runtimeParams:
      let ctype = emitCpuTypeC(p.typ, siteSuffix)
      let sym = "`" & p.name & "`"
      let rhs = if p.typ.kind == ctStruct:
                  "*(" & ctype & "*)&" & sym
                else:
                  "(" & ctype & ")(" & sym & ")"
      ctext.add("    crucibleData." & p.name & " = " & rhs & ";\n")
    ctext.add("    tpc_task* crucibleTaskH = tpc_task_newLoopEnv(\n")
    ctext.add("        crucibleCtx->currentTask, crucibleCtx->currentScope,\n")
    ctext.add("        crucibleTask" & siteSuffix & ", &crucibleData,\n")
    ctext.add("        (int32_t)sizeof(crucibleData), 0,\n")
    ctext.add("        (int32_t)(((`crucibleGrid`.x * `crucibleGrid`.y * `crucibleGrid`.z) + `crucibleWgsPerTask` - 1) / `crucibleWgsPerTask`), 1, true);\n")
    ctext.add("    tpc_schedule(crucibleCtx, crucibleTaskH, false);\n")
    ctext.add("    tpc_syncAll(`crucibleTpHandle`);\n")
    # pool handle: read through Nim first, the emit backtick resolves a declared
    # Nim symbol, C text cannot reach into the engine object
    launchBody.add(nnkLetSection.newTree(nnkIdentDefs.newTree(
      ident"crucibleTpHandle", newEmptyNode(),
      nnkDotExpr.newTree(engine, ident"tp"))))
    launchBody.add(parseStmt("{.emit: \"\"\"" & ctext & "\"\"\" .}"))
  else:
    # coordinate builtins rewrite to loop-variable expressions
    # x-fastest flat index over the grid, workgroup and thread axes
    var vec: array[CirCoordBuiltinKind, array[3, NimNode]]
    vec[cbkThreadgroupPositionInGrid] = [ident"crucibleGx", ident"crucibleGy", ident"crucibleGz"]
    vec[cbkThreadPositionInThreadgroup] = [ident"crucibleBx", ident"crucibleBy", ident"crucibleBz"]
    vec[cbkThreadPositionInGrid] = [
      nnkInfix.newTree(ident"+", nnkInfix.newTree(ident"*", ident"crucibleGx", blkDot("x")), ident"crucibleBx"),
      nnkInfix.newTree(ident"+", nnkInfix.newTree(ident"*", ident"crucibleGy", blkDot("y")), ident"crucibleBy"),
      nnkInfix.newTree(ident"+", nnkInfix.newTree(ident"*", ident"crucibleGz", blkDot("z")), ident"crucibleBz")]
    vec[cbkThreadsPerThreadgroup] = [blkDot("x"), blkDot("y"), blkDot("z")]
    vec[cbkThreadgroupsPerGrid] = [gridDot("x"), gridDot("y"), gridDot("z")]
    let flatIdx = nnkInfix.newTree(ident"+",
      nnkInfix.newTree(ident"+", ident"crucibleBx",
        nnkInfix.newTree(ident"*", ident"crucibleBy", blkDot("x"))),
      nnkInfix.newTree(ident"*", ident"crucibleBz",
        nnkInfix.newTree(ident"*", blkDot("x"), blkDot("y"))))
    let coords = cpuCoordTable(vec, flatIdx)

    # kernel body, inlined, returns become breaks of the per-thread block
    let (body, coordUse) = emitCpuKernelBody(m, siteSuffix, coords)
    if gridN.kind == nnkNilLit and blkN.kind == nnkNilLit and coordUse:
      error("launchCpuFn " & $wrapper[0] & ": kernel reads coordinate builtins and the cpu backend cannot derive the schedule, pass grid/blk in the do block")
    let threadBlock = nnkBlockStmt.newTree(ident"crucibleThread", body)

    proc threadLoops(): NimNode =
      let loopBx = forLoop("crucibleBx", blkDot("x"), threadBlock)
      let loopBy = forLoop("crucibleBy", blkDot("y"), loopBx)
      result = forLoop("crucibleBz", blkDot("z"), loopBy)

    launchBody.add(emitCpuCallees(m, siteSuffix))
    let bodyGx = threadLoops()
    let loopGx = forLoop("crucibleGx", gridDot("x"), bodyGx)
    let loopGy = forLoop("crucibleGy", gridDot("y"), loopGx)
    let loopGz = forLoop("crucibleGz", gridDot("z"), loopGy)
    launchBody.add(nnkStmtList.newTree(loopGz))

  # fresh block scope per launch, two launches in one scope stay isolated
  result = nnkStmtList.newTree(
    nnkBlockStmt.newTree(newEmptyNode(), launchBody))

var launchSiteCounts {.global, compileTime.}: Table[string, int]

macro launchCpuCall(engine: CpuEngine, kernel: typed, gridN, blkN, coarsenN: untyped, siteSuffix: static string, args: varargs[typed]): untyped =
  ## Overload resolution and the compile-time/runtime split, then the handoff to the lowering macro.
  ## Semantic analysis of the generated proc happens before `cpuExecLower` runs, with the wrapper traveling as a typed argument.
  let k = kernel.resolveKernelOverload(args.len)
  let impl = k.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    for j in 0 ..< formals[i].len - 2:
      formalTypes.add(formals[i][^2])
  doAssert args.len <= formalTypes.len,
    "launchCpuFn " & $kernel & ": " & $args.len & " args for " &
    $formalTypes.len & " parameters"

  var statics = nnkBracket.newTree()
  var runtimeArgs: seq[NimNode]
  for i, a in args:
    if isCompileTime(formalTypes[i]):
      statics.add(a)
    else:
      runtimeArgs.add(a)
  let wrapper = buildCpuKernel(k, statics, nnkBracket.newTree(runtimeArgs), siteSuffix)
  result = bindSym"cpuExecLower".newCall(engine, wrapper, gridN, blkN, coarsenN, newLit(siteSuffix))
  for a in runtimeArgs:
    result.add(a)

macro launchCpuFn*(engine: CpuEngine, call: untyped): untyped =
  ## Launches a kernel on the CPU engine with whole values.
  ##
  ## - schedule in a `do:` block, `grid = ...` / `blk = ...`, both optional
  ## - `workgroupsPerTask = K` optional: K adjacent workgroups share one pool task. The serial launch ignores it
  ##
  ##     engine.launchCpuFn axpy(y, 2.5, x) do:
  ##       grid = (1, 1)
  ##       blk = (1, 1)
  ##       workgroupsPerTask = 4
  ## - no schedule: the derived launch
  ##   - coordinate-free kernels run on the (1, 1, 1) x (1, 1, 1) schedule
  ##   - kernels reading coordinate builtins refuse at compile time
  ##   - schedule derivation over tile layouts is unimplemented
  var schedule: NimNode = nil
  var callNode = call
  if callNode[^1].kind == nnkStmtList:
    schedule = callNode[^1]
    callNode = nnkCall.newTree(call[0 ..< call.len - 1])
  var gridN: NimNode = nil   # nil dims reach launchCpuCall as nnkNilLit,
  var blkN: NimNode = nil    # both nil means the derived schedule
  var coarsenN: NimNode = nil  # nil workgroupsPerTask means one workgroup per task
  if not schedule.isNil:
    for a in schedule:
      doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk", "workgroupsPerTask"],
        "launchCpuFn: schedule expects grid = ... / blk = ... / workgroupsPerTask = ..., got " & a.repr
      case $a[0]
      of "grid":
        gridN = a[1]
      of "blk":
        blkN = a[1]
      of "workgroupsPerTask":
        # an integer literal validates here, any other expression hits the runtime guard of the parallel launch
        if a[1].kind == nnkIntLit and a[1].intVal < 1:
          error("launchCpuFn: workgroupsPerTask must be >= 1, got " & $a[1].intVal, a)
        coarsenN = a[1]
  for i in 1 ..< callNode.len:
    doAssert callNode[i].kind notin {nnkExprEqExpr, nnkExprColonExpr},
      "launchCpuFn " & $callNode[0] & ": named arguments are not supported, pass arguments positionally"
  let sitePath = callNode[0].lineInfo[0 ..< callNode[0].lineInfo.rfind('(')]
  let siteBase = splitFile(sitePath).dir / splitFile(sitePath).name
  var siteFile = ""
  for c in siteBase:
    siteFile.add(if c in {'a'..'z', 'A'..'Z', '0'..'9', '_'}: c else: '_')
  # launch symbols and the generated file name carry the sanitized module path
  # plus an occurrence counter. Module files share line/column pairs.
  # Templates reuse their body site
  let siteKey = siteFile & "_L" & $callNode[0].lineInfoObj.line & "C" & $callNode[0].lineInfoObj.column
  let expansion = launchSiteCounts.getOrDefault(siteKey, 0) + 1
  launchSiteCounts[siteKey] = expansion
  let siteSuffix = if expansion > 1: siteKey & "_" & $expansion else: siteKey
  result = bindSym"launchCpuCall".newCall(engine, callNode[0], gridN, blkN, coarsenN, newLit(siteSuffix))
  for i in 1 ..< callNode.len:
    result.add(callNode[i])
