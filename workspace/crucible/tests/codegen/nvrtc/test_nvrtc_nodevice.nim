## NVRTC: calling functions WITHOUT {.device.} set explicitly
##
## Functions defined outside the `cuda` block are auto-annotated with
## `attDevice` in `nim_to_gpu.nim` (~line 832: `fn.pAttributes.incl attDevice`).
## This test verifies the mechanism works for externally-pulled functions.

import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

# ── Functions defined OUTSIDE the cuda block (no {.device.}) ──

proc doubleIt*(x: int32): int32 =  # no {.device.}
  x * 2

proc addScaled*(a, b, scale: int32): int32 =  # no {.device.}
  a + b * scale

# ── Kernel ──

const kernelCode = cuda:
  proc nodeviceKernel(output: ptr UncheckedArray[int32]) {.global.} =
    let tid = int32(blockIdx.x * blockDim.x + threadIdx.x)
    if tid < 4:
      output[tid] = doubleIt(tid + 1)

    let val = addScaled(10, tid, 3)
    output[tid + 4] = val

# ── Test ──────────────────────────────────────────────────────────────

var engine = bkCuda.init()

engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"

var buf: array[8, int32]
engine.run("nodeviceKernel", buf, ())

doAssert buf[0] == 2,  &"doubleIt(1) (no .device.): got {buf[0]}"
doAssert buf[1] == 4,  &"doubleIt(2) (no .device.): got {buf[1]}"
doAssert buf[2] == 6,  &"doubleIt(3) (no .device.): got {buf[2]}"
doAssert buf[3] == 8,  &"doubleIt(4) (no .device.): got {buf[3]}"

# addScaled(10, tid, 3): 10 + tid*3 -> 10, 13, 16, 19
doAssert buf[4] == 10, &"addScaled(tid=0) (no .device.): got {buf[4]}"
doAssert buf[5] == 13, &"addScaled(tid=1) (no .device.): got {buf[5]}"
doAssert buf[6] == 16, &"addScaled(tid=2) (no .device.): got {buf[6]}"
doAssert buf[7] == 19, &"addScaled(tid=3) (no .device.): got {buf[7]}"

echo "  OK — functions WITHOUT {.device.} work (auto-annotated when pulled in)"
