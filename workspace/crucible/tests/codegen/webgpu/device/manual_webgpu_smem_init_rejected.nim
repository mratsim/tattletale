## WebGPU: `{.smem.}` locals refuse an initializer, WGSL workgroup variables are zero-initialized.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/manual_webgpu_smem_init_rejected.nim

import workspace/crucible

const code = webgpu:
  proc k(output: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 1, 1).} =
    var scratch {.smem.}: array[4, uint32] = [1'u32, 2, 3, 4]
    scratch[0] = output[0]
    output[1] = scratch[0]

proc runTest() =
  echo code
  echo "UNREACHABLE — a {.smem.} local with an initializer must be refused at compile time"

when isMainModule:
  runTest()
