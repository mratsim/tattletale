## Metal two-space instantiation: one callee and one struct across two address spaces
##
## Test that pointer address space specializes at instantiation time
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/types/t_ir_spaceInstantiation.nim

import std/[sequtils, sets, strutils]
import workspace/crucible/src/crucible_internal_compiler

proc addOne[T](p: ptr T) {.device.} =
  p[] = p[] + 1.0'f32

type PtrHolder = object
  p: ptr float32

proc distinctCalleeSigs(msl: string, callee: string): seq[(string, string)] =
  ## (mangled name, address-space qualifier) pairs among the `callee` declarations
  ## and definitions in the emitted MSL unit, deduplicated
  for line in msl.split('\n'):
    if callee notin line or "float* p)" notin line:
      continue
    let nameStart = line.find(callee)
    let name = line[nameStart ..< line.find('(', nameStart)]
    let space = if "device float* p)" in line: "device"
                elif "thread float* p)" in line: "thread"
                else: "unqualified"
    if (name, space) notin result:
      result.add (name, space)

proc structDefLines(msl: string): seq[string] =
  ## `struct <Name>{` definition lines of the emitted MSL unit, in emission order
  for line in msl.split('\n'):
    let stripped = line.strip()
    if stripped.startsWith("struct ") and stripped.endsWith("{"):
      result.add stripped

proc collectCallNodes(n: GpuAst, iSym: string, acc: var seq[GpuAst]) =
  ## Every gpuCall whose callee iSym is `iSym`, depth-first pre-order
  if n.isNil:
    return
  if n.kind == gpuCall and not n.cName.symbol.isNil and
      n.cName.symbol.iSym == iSym:
    acc.add n
  for ch in n.items:
    collectCallNodes(ch, iSym, acc)

proc findProcNamed(n: GpuAst, name: string): GpuAst =
  ## First gpuProc whose pName symbol name is exactly `name`
  if n.isNil:
    return nil
  if n.kind == gpuProc and n.pName.symbol.name == name:
    return n
  for ch in n.items:
    let found = findProcNamed(ch, name)
    if not found.isNil:
      return found

proc findProcPrefixed(n: GpuAst, prefix: string): GpuAst =
  ## First gpuProc whose pName symbol name starts with `prefix_`
  ## This is mangledHashName's instantiation-naming shape
  if n.isNil:
    return nil
  if n.kind == gpuProc and n.pName.symbol.name.startsWith(prefix & "_"):
    return n
  for ch in n.items:
    let found = findProcPrefixed(ch, prefix)
    if not found.isNil:
      return found

proc collectCalleeCalls(n: GpuAst, callee: string, acc: var seq[string]) =
  ## iSyms of every `callee` call site in `n`, body order
  if n.isNil:
    return
  if n.kind == gpuCall and n.cName.symbol != nil and
      n.cName.symbol.iSym.startsWith(callee):
    acc.add n.cName.symbol.iSym
  for ch in n:
    collectCalleeCalls(ch, callee, acc)

# ── 1. IR: the two call sites must reference distinct instantiations ──
# Both args are `ptr float32`: the device arg comes from the kernel buffer param,
# the thread arg from a local temp. Pointee storage is the only difference,
# so only a space-aware instantiation can tell the two sites apart.
# The iSyms are observed post-pass, after the bundles rename both call sites.
static:
  let ir = metalIr:
    proc twoSpaceKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var tmp: float32 = 0.0'f32
      addOne(addr outp[0])
      addOne(addr tmp)

  var calls: seq[string]
  ir.collectCalleeCalls("addOne", calls)
  doAssert calls.len == 2, "expected the two call sites, got: " & $calls
  doAssert calls[0] != calls[1],
    "each call site must reference its own space instantiation, both share " &
    calls[0]
  echo "  OK - call sites reference distinct instantiations"

# ── 2. Full Metal pipeline: two space-qualified bodies for the two sites ──
static:
  let msl = metal:
    proc twoSpaceKernelMsl(outp: ptr UncheckedArray[float32]) {.global.} =
      var tmp: float32 = 0.0'f32
      addOne(addr outp[0])
      addOne(addr tmp)

  let sigs = msl.distinctCalleeSigs("addOne")
  doAssert sigs.len == 2,
    "the two call sites need two distinct space-qualified instantiations, got: " &
    $sigs
  doAssert sigs.mapIt(it[1]).toHashSet == ["device", "thread"].toHashSet,
    "one instantiation must serve the device arg, one the thread arg, got: " &
    $sigs

  # the pairing is locked by argument order: the outp[0] site (device storage)
  # must call the device-qualified instantiation, the tmp site the thread one
  # call statements start with the callee name, declarations/definitions with `void`
  let callLines = msl.split('\n').filterIt(it.strip().startsWith("addOne"))
  doAssert callLines.len == 2,
    "both call sites appear in the emitted MSL, got: " & $callLines
  let siteNames = callLines.mapIt(
    it[it.find("addOne") ..< it.find('(', it.find("addOne"))])
  doAssert siteNames == @[sigs.filterIt(it[1] == "device")[0][0],
                          sigs.filterIt(it[1] == "thread")[0][0]],
    "the device-arg site calls the device instantiation, the thread-arg site " &
    "the thread one, got: " & $siteNames
  echo "  OK - two space-qualified callee signatures"

# ── 2b. Hazard: two same-space call sites keep sharing one instantiation ──
static:
  let msl = metal:
    proc sameSpaceKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      addOne(addr outp[0])
      addOne(addr outp[1])

  let sigs = msl.distinctCalleeSigs("addOne")
  doAssert sigs.len == 1,
    "space-keyed instantiation must not split legitimate same-space dedup, " &
    "got: " & $sigs
  echo "  OK - same-space call sites dedup to one instantiation"

# ── 3. Struct surface: two ptr-field constructions across two spaces ──
# Nim-level construction can vary the field space: the device construction
# stores `addr outp[0]`, the thread construction stores `addr tmp`.
# Both compilations below are identical up to construction order, so the emitted
# struct names must match. Type identity must not be picked at render time,
# from whichever construction the collector happens to observe first.
static:
  let mslDeviceFirst = metal:
    proc structKernelDeviceFirst(outp: ptr UncheckedArray[float32]) {.global.} =
      var h1 = PtrHolder(p: addr outp[0])
      var tmp: float32 = 0.0'f32
      var h2 = PtrHolder(p: addr tmp)
      h1.p[] = 1.0'f32
      h2.p[] = 2.0'f32
      outp[1] = h1.p[]
      outp[2] = h2.p[]

  let mslThreadFirst = metal:
    proc structKernelThreadFirst(outp: ptr UncheckedArray[float32]) {.global.} =
      var tmp: float32 = 0.0'f32
      var h2 = PtrHolder(p: addr tmp)
      var h1 = PtrHolder(p: addr outp[0])
      h2.p[] = 2.0'f32
      h1.p[] = 1.0'f32
      outp[1] = h1.p[]
      outp[2] = h2.p[]

  let defsDeviceFirst = mslDeviceFirst.structDefLines
  let defsThreadFirst = mslThreadFirst.structDefLines

  doAssert defsDeviceFirst.len == 2,
    "both space variants of the struct must be emitted as distinct defs, got: " &
    $defsDeviceFirst
  doAssert "device float* p;" in mslDeviceFirst and
           "thread float* p;" in mslDeviceFirst,
    "the two struct defs must carry distinct address-space qualifiers, got: " &
    $defsDeviceFirst
  doAssert defsDeviceFirst == defsThreadFirst,
    "struct def names must not depend on construction observation order, got: " &
    $defsDeviceFirst & " vs " & $defsThreadFirst
  echo "  OK - struct defs are order-independent"

# ── 3b. Two same-space constructions share one instantiated def ──
static:
  let msl = metal:
    proc structKernelSameSpace(outp: ptr UncheckedArray[float32]) {.global.} =
      var tmp: float32 = 0.0'f32
      var h1 = PtrHolder(p: addr tmp)
      var h2 = PtrHolder(p: addr tmp)
      h1.p[] = 1.0'f32
      h2.p[] = 2.0'f32
      outp[1] = h1.p[]
      outp[2] = h2.p[]

  let defs = msl.structDefLines
  doAssert defs.len == 2,
    "the declared def and one instantiated def serve both same-space sites, got: " & $defs
  doAssert defs.countIt(it.startsWith("struct PtrHolder_")) == 1,
    "same-space constructions must dedup to one instantiated def, got: " & $defs
  echo "  OK - same-space struct constructions dedup to one def"

# ── 4. Nested calls instantiate recursively, declared fns stay untouched ──
static:
  let ir = metalIr:
    var gbuf: float32
    proc leaf(data: openArray[float32]) {.device.} =
      gbuf = data[0]
    proc middle(data: openArray[float32], count: var int32) {.device.} =
      leaf(data)
      count = count + 1
    proc nestKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var tmpArr: array[4, float32]
      var n: int32 = 0
      middle(toOpenArray(outp, 0, 3), n)
      middle(toOpenArray(tmpArr, 0, 3), n)

  # two sites on middle: the device span site gets its own instantiation
  # the thread span site binds the declared middle fn unchanged
  # the var count parameter multiplies nothing across the sites
  var middleCalls: seq[string]
  ir.collectCalleeCalls("middle", middleCalls)
  doAssert middleCalls.len == 2 and middleCalls[0] != middleCalls[1],
    "the two span sites reference distinct middle fns, got: " & $middleCalls
  let declaredMiddle = ir.findProcNamed("middle")
  doAssert not declaredMiddle.isNil,
    "the thread span site binds the declared middle"
  doAssert declaredMiddle.pParams[0].typ.kind == gtPtr and
    declaredMiddle.pParams[0].typ.addressSpace == asRMEM,
    "the declared middle's split ptr keeps its birth space, got " &
    $declaredMiddle.pParams[0].typ.addressSpace

  # the instantiated middle specializes its split ptr to the device span
  let instMiddle = ir.findProcPrefixed("middle")
  doAssert not instMiddle.isNil, "the device span site got a middle instantiation"
  doAssert instMiddle.pParams[0].typ.kind == gtPtr and
    instMiddle.pParams[0].typ.addressSpace == asDevice,
    "the instantiated middle's split ptr specializes to device, got " &
    $instMiddle.pParams[0].typ.addressSpace

  # the instantiated body's leaf call instantiates leaf in turn
  # the forwarded argument is the instantiation's own fresh device symbol
  var leafCalls: seq[string]
  instMiddle.pBody.collectCalleeCalls("leaf", leafCalls)
  doAssert leafCalls.len == 1,
    "the instantiated middle calls leaf once, got: " & $leafCalls
  let declaredLeaf = ir.findProcNamed("leaf")
  doAssert not declaredLeaf.isNil and
    leafCalls[0] != declaredLeaf.pName.symbol.iSym,
    "the nested leaf call references its own instantiation, not the declared leaf"
  let instLeaf = ir.findProcPrefixed("leaf")
  doAssert not instLeaf.isNil and
    instLeaf.pParams[0].typ.addressSpace == asDevice,
    "the nested leaf instantiation specializes to device, got " &
    $instLeaf.pParams[0].typ.addressSpace
  var leafArgs: seq[GpuAst]
  collectCallNodes(instMiddle.pBody, leafCalls[0], leafArgs)
  doAssert leafArgs.len == 1 and leafArgs[0].cArgs[0].kind == gpuIdent and
    leafArgs[0].cArgs[0].symbol.typ.addressSpace == asDevice,
    "the forwarded span arg is the instantiation's fresh device symbol, got " &
    $leafArgs[0].cArgs[0].symbol.typ.addressSpace
  echo "  OK - nested instantiation recurses, declared fns stay untouched"

echo "t_ir_spaceInstantiation: all static assertions passed"
