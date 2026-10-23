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

proc collectCalleeCalls(n: GpuAst, callee: string, acc: var seq[string]) =
  ## iSyms of every `callee` call site in `n`, body order
  if n.isNil:
    return
  if n.kind == gpuCall and n.cName.symbol != nil and
      n.cName.symbol.iSym.startsWith(callee):
    acc.add n.cName.symbol.iSym
  for ch in n:
    collectCalleeCalls(ch, callee, acc)

# ── 1. Frontend: the two call sites must reference distinct instantiations ──
# Both args are `ptr float32`: the device arg comes from the kernel buffer param,
# the thread arg from a local temp. Pointee storage is the only difference,
# so only a space-aware instantiation can tell the two sites apart.
static:
  let ir = toCrucibleAst:
    proc twoSpaceKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var tmp: float32 = 0.0'f32
      addOne(addr outp[0])
      addOne(addr tmp)

  var calls: seq[string]
  ir.statements[0].collectCalleeCalls("addOne", calls)
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
  let callLines = msl.split('\n').filterIt("addOne((" in it)
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

echo "t_ir_spaceInstantiation: all static assertions passed"
