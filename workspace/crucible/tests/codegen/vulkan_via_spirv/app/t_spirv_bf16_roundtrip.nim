## SPIR-V bfloat16: u16 storage, f32 compute, spirv-val checked, exact on MoltenVK
##
## Test that kernels over `bfloat16` buffers emit binary SPIR-V accepted by
## `spirv-val --target-env vulkan1.4` and run exact on the Vulkan device:
## the buffer element stores as its u16 bit pattern, loads promote to f32,
## stores truncate f32 back with round-to-nearest-even.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_bf16_roundtrip.nim

import std/[os, osproc, strutils]
import workspace/crucible

proc bf16TestValues(): array[8, float32] =
  # 0: bf16-representable, exact promotion
  # 1: half-ulp tie below an even mantissa, RNE rounds down to 1.0
  # 2: half-ulp tie above an odd mantissa, RNE rounds up to 1.015625
  # 3: sub-tie with sticky bits, rounds down
  # 4: sub-tie with sticky bits crossing the rounding boundary, rounds up
  # 5: a negative bf16 value
  # 6: positive infinity
  # 7: a quiet NaN
  [1.0'f32, 1.00390625'f32, 1.01171875'f32, 1.001953125'f32, 1.009765625'f32,
   -2.5'f32, Inf, NaN]

const promoteSpv = spirv:
  proc promoteKernel(outF: ptr UncheckedArray[float32], x: ptr UncheckedArray[bfloat16], n: int32) {.global.} =
    let i = thread_position_in_grid.x
    if i < n:
      outF[i] = x[i].to(float32)

const demoteSpv = spirv:
  proc demoteKernel(outB: ptr UncheckedArray[bfloat16], x: ptr UncheckedArray[float32], n: int32) {.global.} =
    let i = thread_position_in_grid.x
    if i < n:
      outB[i] = x[i].to(bfloat16)

const addSpv = spirv:
  proc bf16AddKernel(outB: ptr UncheckedArray[bfloat16], a, b: ptr UncheckedArray[bfloat16], n: int32) {.global.} =
    let i = thread_position_in_grid.x
    if i < n:
      outB[i] = (a[i].to(float32) + b[i].to(float32)).to(bfloat16)

proc validateSpv(words: seq[uint32], name: string) =
  doAssert words.len > 0, "`spirv:` macro emitted no words for " & name
  doAssert words[0] == 0x07230203'u32, "SPIR-V magic number missing for " & name
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / (name & ".spv")
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr words[0], words.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the emitted module " & name & ":\n" & valOut

proc runTest() =
  validateSpv(promoteSpv, "t_spirv_bf16_promote")
  validateSpv(demoteSpv, "t_spirv_bf16_demote")
  validateSpv(addSpv, "t_spirv_bf16_add")

  let vals = bf16TestValues()

  block promotion:
    # bf16 storage -> f32 compute: the device reads the u16 bit pattern and widens exactly.
    # Readback equals the host widening.
    var host = newSeq[bfloat16](vals.len)
    for i in 0 ..< vals.len:
      host[i] = vals[i].to(bfloat16)
    var engine = newEngine(runVulkan)
    engine.ingest(promoteSpv)
    let xDev = engine.allocDevice(bfloat16, host.len)
    let oDev = engine.allocDevice(float32, host.len)
    engine.copyToDeviceFromHost(xDev, host)
    engine.launchKernel("promoteKernel", oDev, (xDev, int32(host.len)))
    var got: array[8, float32]
    engine.copyToHostFromDevice(got.toOpenArray(0, host.len - 1), oDev)
    for i in 0 ..< vals.len:
      let want = host[i].to(float32)
      doAssert cast[uint32](got[i]) == cast[uint32](want),
        "promotion " & $i & " got " & $got[i] & ", want " & $want
    echo "  promotion: bf16 storage reads back exact f32 (incl. inf/NaN)"

  block truncation:
    # f32 -> bf16 storage: the device truncates with round-to-nearest-even.
    # Readback bits equal the host toBf16 bits.
    var engine = newEngine(runVulkan)
    engine.ingest(demoteSpv)
    let xDev = engine.allocDevice(float32, vals.len)
    let oDev = engine.allocDevice(bfloat16, vals.len)
    engine.copyToDeviceFromHost(xDev, vals)
    engine.launchKernel("demoteKernel", oDev, (xDev, int32(vals.len)))
    var got: array[8, bfloat16]
    engine.copyToHostFromDevice(got.toOpenArray(0, vals.len - 1), oDev)
    for i in 0 ..< vals.len:
      let want = vals[i].to(bfloat16)
      doAssert cast[uint16](got[i]) == cast[uint16](want),
        "truncation " & $i & " got bits 0x" & toHex(cast[uint16](got[i]), 4) &
        ", want 0x" & toHex(cast[uint16](want), 4)
    # Hand-checked rounding decisions over the tie values.
    doAssert cast[uint16](got[1]) == 0x3F80'u16, "half-ulp below 1.0 rounds down to 1.0"
    doAssert cast[uint16](got[2]) == 0x3F82'u16, "half-ulp above 1.0078125 rounds up to 1.015625"
    doAssert cast[uint16](got[6]) == 0x7F80'u16, "infinity keeps its bit pattern"
    doAssert cast[uint16](got[7]) == 0x7FC0'u16, "NaN quietens to the canonical 0x7FC0"
    echo "  truncation: f32 stores back as RNE u16 bit patterns (incl. inf/NaN)"

  block bf16Add:
    # f32-compute add over bf16 buffers, exact against the host arithmetic.
    var ah, bh: seq[bfloat16]
    for i in 0 ..< 8:
      ah.add vals[i].to(bfloat16)
      bh.add float32(i + 1).to(bfloat16)
    var engine = newEngine(runVulkan)
    engine.ingest(addSpv)
    let aDev = engine.allocDevice(bfloat16, ah.len)
    let bDev = engine.allocDevice(bfloat16, bh.len)
    let oDev = engine.allocDevice(bfloat16, ah.len)
    engine.copyToDeviceFromHost(aDev, ah)
    engine.copyToDeviceFromHost(bDev, bh)
    engine.launchKernel("bf16AddKernel", oDev, (aDev, bDev, int32(ah.len)))
    var got: array[8, bfloat16]
    engine.copyToHostFromDevice(got.toOpenArray(0, ah.len - 1), oDev)
    for i in 0 ..< ah.len:
      let want = (ah[i].to(float32) + bh[i].to(float32)).to(bfloat16)
      doAssert cast[uint16](got[i]) == cast[uint16](want),
        "add " & $i & " got bits 0x" & toHex(cast[uint16](got[i]), 4) &
        ", want 0x" & toHex(cast[uint16](want), 4)
    echo "  add: f32 compute over bf16 storage, exact readback"

when isMainModule:
  runTest()
  echo "t_spirv_bf16_roundtrip: spirv-val accepted all modules, values exact on MoltenVK"
