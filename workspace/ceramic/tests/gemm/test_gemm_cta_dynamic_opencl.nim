## Unified GEMM entry: OpenCL compile-gate + NVIDIA-device-gated run.
##
## The unified manual_gemm_cta_dynamic entry expands its kernel for OpenCL
## at Nim compile time. This test pins the generated OpenCL text: the flat global id
## lowers to `get_global_id(0)`, and the gemm_cta barriers lower to `barrier(CLK_LOCAL_MEM_FENCE)`.
## The kernel executes only on NVIDIA's OpenCL compiler (the mma.sync inline PTX),
## so the run is gated on the device vendor. On a non-NVIDIA machine, the gate
## reports the skip and the Linux box runs the same entry via manual_gemm_cta_dynamic.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/gemm --nimcache:nimcache/tests/gemm \
##     workspace/ceramic/tests/gemm/test_gemm_cta_dynamic_opencl.nim

import std/strutils
import workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The flat global id: canonical thread_position_in_grid.x lowers to get_global_id(0) with the OpenCL cast form.
  doAssert "(int)get_global_id(0)" in kernelCodeOpenCL,
    "OpenCL global-id cast missing:\n" & kernelCodeOpenCL
  # The gemm_cta barriers lower to barrier(CLK_LOCAL_MEM_FENCE).
  doAssert kernelCodeOpenCL.count("barrier(CLK_LOCAL_MEM_FENCE)") >= 2,
    "OpenCL barriers missing:\n" & kernelCodeOpenCL
  echo "  OK — unified GEMM OpenCL source inspect (compile + grep)"

  var engine = bkOpenCL.init(kernelCodeOpenCL)
  if "NVIDIA" notin engine.deviceName():
    echo "  SKIP — OpenCL execution needs NVIDIA's OpenCL compiler (mma.sync asm); device: " &
      engine.deviceName()
    return

  # The execution gate: full harness on an NVIDIA OpenCL device.
  const kView = 64
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     64, 32, 32, 80, 48, 80, kView)
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     64, 32, 64, 64, 32, 64, kView)
  echo "  OK — unified GEMM OpenCL execution (NVIDIA device)"

when isMainModule:
  runTest()
