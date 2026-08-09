## Phase 2 GPU probe — mma.sync m16n8k8 tf32, raw emission.
##
## The riskiest unknown, isolated: can the DSL emit a real tensor-core
## instruction with register operands, and does it run on sm_120?
##
## Strategy: every thread fills its A (4×u32), B (2×u32) and C (4×f32)
## fragment registers with KNOWN values, executes one
##   mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
## and writes D back. With uniform register values the result is
## layout-independent: D = (k-sum of A·B) + C = 8·a·b + c for every
## element. The fragment-layout mapping itself is validated by the
## Phase 1 dump; this probe validates instruction text, register
## counts, PTX operand constraints, tf32 interpretation, and sm_120
## availability in one shot.
##
## tf32 note: register bits are interpreted as tf32 (top 19 bits of
## f32) — no cvt needed for integer-valued floats; the host reference
## truncates the mantissa to 10 bits to match.
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm/test_mma_sync_probe.nim

import std/strformat
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc mmaSyncProbe(outD: ptr UncheckedArray[float32]) {.global.} =
    ## Uniform-value smoke: A=B=2.0, C=5.0 → D = 8·2·2 + 5 = 37.
    ## tf32 register bits: 2.0f = 0x40000000 (integer-valued floats need
    ## no mantissa rounding — the tf32 interpretation keeps the top 19
    ## bits, which for values < 2^11 with zero mantissa are exact).
    let t = int(threadIdx.x)
    let aVal = 0x40000000'u32   # 2.0f bit pattern
    let bVal = 0x40000000'u32
    var d0 = 5.0'f32
    var d1 = 5.0'f32
    var d2 = 5.0'f32
    var d3 = 5.0'f32
    # One m16n8k8 tf32 mma.sync with f32 accumulator, inline (locals are
    # lvalues; var-proc params would emit as pointers and break the "f"
    # constraint). A: 4×u32, B: 2×u32, C/D: 4×f32.
    asm "\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\" : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3) : \"r\"(aVal), \"r\"(aVal), \"r\"(aVal), \"r\"(aVal), \"r\"(bVal), \"r\"(bVal)"
    outD[t * 4 + 0] = d0
    outD[t * 4 + 1] = d1
    outD[t * 4 + 2] = d2
    outD[t * 4 + 3] = d3

when isMainModule:
  var outD = newSeq[float32](32 * 4)
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()
  echo "PTX: ", nv.ptx.len, " bytes"
  nv.execute("mmaSyncProbe", dim3(1), dim3(32), outD, ())

  # expected: D = 8·2·2 + 5 = 37 for every element of every thread
  var failures = 0
  for t in 0 ..< 32:
    for i in 0 ..< 4:
      if outD[t * 4 + i] != 37.0'f32:
        failures.inc
        if failures <= 3:
          echo &"  t{t} d{i} = {outD[t * 4 + i]} (expected 37)"
  doAssert failures == 0, &"{failures} mismatches"
  echo "  OK — mma.sync m16n8k8 tf32 runs on sm_120, register contract correct (A4/B2/C4→D4)"
