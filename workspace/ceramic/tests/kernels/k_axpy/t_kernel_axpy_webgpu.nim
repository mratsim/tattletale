## Tile axpy kernel on WebGPU
##
## Test that the tile axpy kernel lowers through WGSL and computes
## y = 2.5*x + y over five layout variants against the host reference
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_axpy/t_kernel_axpy_webgpu.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_axpy/kernel_axpy_template

proc main =
  var engine = newEngine(runWebGPU)
  run_axpy_tests(engine, moduleName = "test_k_axpy_webgpu")

when isMainModule:
  main()
