# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[compilesettings, macros, md5, os, strutils, tables]

import workspace/crucible/src/codegen/cpu_compiler
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/engines/cpu_api
# ═══════════════════════════════════════════════════════════════════════
# ▸ Launch-site helpers
# ═══════════════════════════════════════════════════════════════════════

func isCompileTime(ftype: NimNode): bool =
  ftype.kind == nnkStaticTy or
    (ftype.kind in {nnkIdent, nnkSym} and ftype.eqIdent("typedesc")) or
    (ftype.kind in {nnkCommand, nnkBracketExpr, nnkCall} and
     $ftype[0] in ["static", "typedesc"])

func formalCounts(candFormals: NimNode): tuple[total, required: int] =
  for i in 1 ..< candFormals.len:
    let f = candFormals[i]
    result.total += f.len - 2
    if f[^1].kind == nnkEmpty:
      result.required += f.len - 2

func runtimeFormalsAccept(candImpl: NimNode, args: NimNode): bool =
  let formals = candImpl[3]
  var argIdx = 0
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      if argIdx >= args.len:
        return true
      if not isCompileTime(f[^2]) and
          not sameType(args[argIdx].getTypeInst(), f[^2]):
        return false
      argIdx.inc
  result = true

func candidateSignatures(cands: seq[NimNode]): string =
  for cand in cands:
    let formals = cand.getImpl()[3]
    var types: seq[string]
    for i in 1 ..< formals.len:
      for j in 0 ..< formals[i].len - 2:
        types.add(formals[i][^2].repr)
    result.add("  " & $cand & "(" & types.join(", ") & ")\n")

func resolveKernelOverload(kernel: NimNode, args: NimNode): NimNode =
  if kernel.kind notin {nnkOpenSymChoice, nnkClosedSymChoice}:
    return kernel
  var arityMatches: seq[NimNode]
  for cand in kernel:
    let candImpl = cand.getImpl()
    if candImpl.kind != nnkProcDef:
      continue
    let (total, required) = formalCounts(candImpl[3])
    if required <= args.len and args.len <= total:
      arityMatches.add(cand)
  if arityMatches.len > 1:
    var typeMatches: seq[NimNode]
    for cand in arityMatches:
      if runtimeFormalsAccept(cand.getImpl(), args):
        typeMatches.add(cand)
    if typeMatches.len == 1:
      return typeMatches[0]
  doAssert arityMatches.len == 1,
    "kernel launch " & $kernel & ": " & $arityMatches.len & " overloads take " &
    $args.len & " arguments, the runtime argument types must select exactly one," &
    " candidates:\n" & candidateSignatures(arityMatches)
  arityMatches[0]

proc assertDimsWithinInt32(grid, blk: Dim3) =
  for (kind, dim) in [("grid", grid.x), ("grid", grid.y), ("grid", grid.z),
                      ("blk", blk.x), ("blk", blk.y), ("blk", blk.z)]:
    doAssert dim <= high(int32),
      "launchCpuFn: " & kind & " dim " & $dim & " exceeds int32, the emitted C launch stores each dim as int32"
  # every dim passed the int32 bound above, so each int64 product below
  # cannot wrap and the guard stays correct with overflow checks off
  let plane = int64(grid.x) * int64(grid.y)
  doAssert grid.z == 0 or plane <= int64(high(int32)) div grid.z,
    "launchCpuFn: the grid workgroup total " & $grid.x & "*" & $grid.y & "*" & $grid.z &
    " exceeds int32, the emitted C launch computes the workgroup total in int32"
  for (axis, g, b) in [("x", grid.x, blk.x), ("y", grid.y, blk.y), ("z", grid.z, blk.z)]:
    doAssert int64(g) * int64(b) <= high(int32),
      "launchCpuFn: the " & axis & " axis schedule " & $g & "*" & $b &
      " exceeds int32, the emitted C launch computes thread_position_in_grid in int32"
  let blockThreads = int64(blk.x) * int64(blk.y) * int64(blk.z)
  doAssert blockThreads <= high(int32),
    "launchCpuFn: the workgroup thread total " & $blk.x & "*" & $blk.y & "*" & $blk.z &
    " exceeds int32, the emitted C launch computes thread_index_in_threadgroup in int32"

# ═══════════════════════════════════════════════════════════════════════
# ▸ Kernel wrapper
# ═══════════════════════════════════════════════════════════════════════

proc isDeviceMemArg(t: NimNode): bool =
  t.kind == nnkBracketExpr and t[0].eqIdent("DeviceMem")

proc deviceMemFieldNames(t: NimNode): seq[string] =
  let impl = t.getTypeImpl()
  case impl.kind
  of nnkObjectTy:
    let rec = impl[^1]
    if rec.kind == nnkRecList:
      for f in rec:
        f.expectKind(nnkIdentDefs)
        if isDeviceMemArg(f[^2]):
          for j in 0 ..< f.len - 2:
            result.add($f[j])
  of nnkTupleTy:
    for i, f in impl:
      if isDeviceMemArg(f[^2]):
        result.add("field" & $i)
  else:
    discard

proc pointeeDeviceMemRefusal(v: NimNode): string =
  let t = v.getTypeInst()
  if t.kind != nnkPtrTy:
    return ""
  var pointee = t[0]
  if pointee.kind == nnkBracketExpr and pointee[0].eqIdent("UncheckedArray"):
    pointee = pointee[1]
  let fields = pointee.deviceMemFieldNames()
  if fields.len == 0:
    return ""
  let names = if fields.len == 1: fields[0] else: fields.join(", ")
  return "points at " & pointee.repr & " which holds DeviceMem field " & names

proc buildCpuKernel(kernel: NimNode, statics, values: NimNode, siteSuffix: string): NimNode =
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
      let pointeeRefusal = v.pointeeDeviceMemRefusal()
      if pointeeRefusal.len > 0:
        error("launchCpuFn " & $kernel & ": parameter " & $formalNames[i].repr & " " &
          pointeeRefusal & ", a GPU runtime concept, the cpu engine takes host values")
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

var launchGensymCounter {.global, compileTime.}: int
  ## Per-launch suffix for the generated launch counters (`crucibleGx_7`).
  ## A counter shares the C scope of the inlined kernel body.
  ## No kernel-local name can collide with a counter.

macro cpuExecLower(engine: CpuEngine, wrapper: typed, gridN, blkN, coarsenN: untyped, siteSuffix: static string, args: varargs[typed]): untyped =
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

  var hoists = nnkStmtList.newTree()
  for i, nameNode in runtimeNames:
    let name = $nameNode
    let arg = args[i]
    if not (arg.kind == nnkIdent and $arg == name):
      hoists.add(nnkVarSection.newTree(
        nnkIdentDefs.newTree(ident(name), newEmptyNode(), arg)))

  var launchBody = nnkStmtList.newTree(gridDecl, blkDecl,
    nnkCall.newTree(bindSym"assertDimsWithinInt32", gridVar, blkVar),
    hoists)
  let serial = not compileOption("threads")
  if not serial:
    # envSize in the pooled tail is the data-struct size cast to int32_t,
    # a struct at or above 2^31 bytes wraps the cast
    # 64 bytes cover the counters, the grid and block dims and member padding
    var dataBytes: NimNode = newLit(64)
    for nameNode in runtimeNames:
      dataBytes = nnkInfix.newTree(ident"+", dataBytes,
        nnkCall.newTree(bindSym"sizeof", ident($nameNode)))
    let refusalMsg = nnkInfix.newTree(ident"&",
      newLit("launchCpuFn " & $wrapper[0] & ": the launch data struct needs "),
      nnkInfix.newTree(ident"&",
        nnkCall.newTree(bindSym"$", dataBytes),
        newLit(" bytes, the pooled launch stores the struct size in an int32 envSize, keep the runtime parameters under 2 GiB")))
    launchBody.add(quote do:
      static:
        doAssert `dataBytes` <= high(int32), `refusalMsg`)
  let gridReads = ["tpc_d_->crucibleGrid[0]", "tpc_d_->crucibleGrid[1]",
                   "tpc_d_->crucibleGrid[2]"]
  let blkReads = ["tpc_d_->crucibleBlk[0]", "tpc_d_->crucibleBlk[1]",
                  "tpc_d_->crucibleBlk[2]"]
  launchGensymCounter.inc
  let counterTag = $launchGensymCounter
  var vecC: array[CirCoordBuiltinKind, array[3, string]]
  vecC[cbkThreadgroupPositionInGrid] = [
    "crucibleGx_" & counterTag, "crucibleGy_" & counterTag, "crucibleGz_" & counterTag]
  vecC[cbkThreadPositionInThreadgroup] = [
    "crucibleBx_" & counterTag, "crucibleBy_" & counterTag, "crucibleBz_" & counterTag]
  vecC[cbkThreadPositionInGrid] = [
    "(crucibleGx_" & counterTag & " * " & blkReads[0] & " + crucibleBx_" & counterTag & ")",
    "(crucibleGy_" & counterTag & " * " & blkReads[1] & " + crucibleBy_" & counterTag & ")",
    "(crucibleGz_" & counterTag & " * " & blkReads[2] & " + crucibleBz_" & counterTag & ")"]
  vecC[cbkThreadsPerThreadgroup] = blkReads
  vecC[cbkThreadgroupsPerGrid] = gridReads
  let flatC = "(crucibleBx_" & counterTag & " + crucibleBy_" & counterTag & " * " & blkReads[0] &
              " + crucibleBz_" & counterTag & " * " & blkReads[0] & " * " & blkReads[1] & ")"
  let coordsC = cpuCoordTableC(vecC, flatC)

  let runtimeParams = kernelRuntimeParams(m)
  let (typesAndData, funcsC, coordUseC) = emitCpuWorkgroupLaunchC(
      m, siteSuffix, counterTag, coordsC, runtimeParams, serial)
  if gridN.kind == nnkNilLit and blkN.kind == nnkNilLit and coordUseC:
    error("launchCpuFn " & $wrapper[0] & ": kernel reads coordinate builtins and the cpu backend cannot derive the schedule, pass grid/blk in the do block")
  let calleesC = emitCpuCalleesC(m, siteSuffix)

  var unitC = typesAndData
  if serial:
    # stddef.h defines NULL for the nil literals the printer emits,
    # pooled units reach it through the threadpool force-includes
    unitC = "#include <stddef.h>\n#include <stdint.h>\n#include <stdbool.h>\n" & unitC
  if calleesC.len > 0:
    unitC.add(calleesC & "\n")
  unitC.add(funcsC)
  let genDir = querySetting(SingleValueSetting.nimcacheDir) / "crucibleGenerated"
  createDir(genDir)
  # NAME_MAX is 255 and the site suffix embeds the absolute module path:
  # long file names shrink to a bounded head plus an md5 hash
  let genStem = if siteSuffix.len <= 128: siteSuffix
                else: siteSuffix[0 ..< 96] & "_" & $toMD5(siteSuffix)
  let genPath = genDir / ("crucibleLaunch_" & genStem & ".c")
  writeFile(genPath, unitC)
  launchBody.add(parseStmt("{.compile: \"" & genPath & "\".}"))

  var ctext = typesAndData
  # inside emit text, a backtick-quoted Nim symbol emits as its C name
  if serial:
    ctext.add(emitCpuSerialTailC(siteSuffix, runtimeParams))
  else:
    launchBody.add(nnkLetSection.newTree(
      nnkIdentDefs.newTree(ident"crucibleWgsPerTask", newEmptyNode(),
        nnkCall.newTree(bindSym"int32", coarsenNeff))))
    launchBody.add(nnkCall.newTree(bindSym"doAssert",
      nnkInfix.newTree(ident">=", ident"crucibleWgsPerTask", newLit(1'i32)),
      newLit("launchCpuFn: workgroupsPerTask must be >= 1")))
    ctext.add("    extern void crucibleTask" & siteSuffix & "(void*);\n")
    ctext.add("    tpc_worker_context* crucibleCtx = tpc_workerContext();\n")
    ctext.add("    struct crucibleData" & siteSuffix & " crucibleData;\n")
    ctext.add(emitCpuDataFillC(siteSuffix, runtimeParams))
    ctext.add("    crucibleData.crucibleWgsPerTask = (int32_t)`crucibleWgsPerTask`;\n")
    ctext.add("    tpc_task* crucibleTaskH = tpc_task_newLoopEnv(\n")
    ctext.add("        crucibleCtx->currentTask, crucibleCtx->currentScope,\n")
    ctext.add("        crucibleTask" & siteSuffix & ", &crucibleData,\n")
    ctext.add("        (int32_t)sizeof(crucibleData), 0,\n")
    ctext.add("        (int32_t)(((`crucibleGrid`.x * `crucibleGrid`.y * `crucibleGrid`.z) + `crucibleWgsPerTask` - 1) / `crucibleWgsPerTask`), 1, true);\n")
    ctext.add("    tpc_schedule(crucibleCtx, crucibleTaskH, false);\n")
    ctext.add("    tpc_syncAll(`crucibleTpHandle`);\n")
    launchBody.add(nnkLetSection.newTree(nnkIdentDefs.newTree(
      ident"crucibleTpHandle", newEmptyNode(),
      nnkDotExpr.newTree(engine, ident"tp"))))
  launchBody.add(parseStmt("{.emit: \"\"\"" & ctext & "\"\"\" .}"))

  result = nnkStmtList.newTree(
    nnkBlockStmt.newTree(newEmptyNode(), launchBody))

var launchSiteCounts {.global, compileTime.}: Table[string, int]

macro launchCpuCall(engine: CpuEngine, kernel: typed, gridN, blkN, coarsenN: untyped, siteSuffix: static string, args: varargs[typed]): untyped =
  let k = kernel.resolveKernelOverload(args)
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
  ## - the schedule is an optional `do:` block, one assignment per line:
  ##
  ##     engine.launchCpuFn axpy(y, 2.5, x) do:
  ##       grid = (2, 1, 1)
  ##       blk = (32, 1, 1)
  ##       workgroupsPerTask = 4
  ##
  ## - `grid`, `blk` and `workgroupsPerTask` are each optional
  var schedule: NimNode = nil
  var callNode = call
  var gridN: NimNode = nil
  var blkN: NimNode = nil
  var coarsenN: NimNode = nil
  if callNode[^1].kind == nnkStmtList:
    schedule = callNode[^1]
    callNode = nnkCall.newTree(call[0 ..< call.len - 1])
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
        if a[1].kind == nnkIntLit and a[1].intVal < 1:
          error("launchCpuFn: workgroupsPerTask must be >= 1, got " & $a[1].intVal, a)
        if not compileOption("threads") and a[1].kind != nnkIntLit:
          error("launchCpuFn: the serial launch has no pool tasks, workgroupsPerTask must be an integer literal (it is ignored), got the expression " & a[1].repr, a)
        coarsenN = a[1]
  for i in 1 ..< callNode.len:
    doAssert callNode[i].kind notin {nnkExprEqExpr, nnkExprColonExpr},
      "launchCpuFn " & $callNode[0] & ": named arguments are not supported, pass arguments positionally"
  let sitePath = callNode[0].lineInfo[0 ..< callNode[0].lineInfo.rfind('(')]
  let siteBase = splitFile(sitePath).dir / splitFile(sitePath).name
  var siteFile = ""
  for c in siteBase:
    siteFile.add(if c in {'a'..'z', 'A'..'Z', '0'..'9', '_'}: c else: '_')
  let siteKey = siteFile & "_L" & $callNode[0].lineInfoObj.line & "C" & $callNode[0].lineInfoObj.column
  let expansion = launchSiteCounts.getOrDefault(siteKey, 0) + 1
  launchSiteCounts[siteKey] = expansion
  let siteSuffix = if expansion > 1: siteKey & "_" & $expansion else: siteKey
  result = bindSym"launchCpuCall".newCall(engine, callNode[0], gridN, blkN, coarsenN, newLit(siteSuffix))
  for i in 1 ..< callNode.len:
    result.add(callNode[i])
