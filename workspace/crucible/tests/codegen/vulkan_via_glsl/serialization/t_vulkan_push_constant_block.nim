## Vulkan push-constant block layout, host model vs the compiled module
##
## Test that the host layout model reproduces the byte offsets glslang
## assigns to the GEMM's tensor-view push-constant block.
## Every tensor view occupies 32 bytes. The whole block totals 144 bytes.
##
## Expected values are measured, not derived: they are the OpMemberDecorate
## offsets of the KernelParams block in the SPIR-V module glslang produced
## for the same launch (D@0, alpha@32, A@40, B@72, beta@104, C@112).
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/serialization/t_vulkan_push_constant_block.nim

import workspace/crucible
import workspace/crucible/src/runtime/launch/launch_serialization
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/runtime/backends/runtime_utils

type
  TensorViewLayout = object
    shape: (int, int)
    stride: (int, int)

  GemmTensorView = object
    ## Device-side arg shape of the GEMM test's bridge tensor:
    ## buffer handle, element offset, runtime shape and strides.
    data: DeviceMem[VulkanBuffer, float32]
    offset: int
    layout: TensorViewLayout

func pushBlockBytes(backend: static CodegenKind): int {.compileTime.} =
  ## Push-constant bytes of the GEMM launch, arguments placed at device
  ## alignment in launch order: D, alpha, A, B, beta and finally C.
  const view = computeDeviceLayout(GemmTensorView, backend)
  const scalar = computeDeviceLayout(float32, backend)
  var off = 0
  off = off.round_multiple_up(view.align) + view.size
  off = off.round_multiple_up(scalar.align) + scalar.size
  off = off.round_multiple_up(view.align) + view.size
  off = off.round_multiple_up(view.align) + view.size
  off = off.round_multiple_up(scalar.align) + scalar.size
  off = off.round_multiple_up(view.align) + view.size
  off

static:
  doAssert computeDeviceLayout(GemmTensorView, cgGlsl) == (size: 32, align: 8),
    "the tensor view must pack as glslang lays it out: 8-byte address, 4-byte offset, 16-byte layout, padded to 8"
  doAssert pushBlockBytes(cgGlsl) == 144,
    "the GEMM push block must total 144 bytes: glslang assigns D@0, alpha@32, A@40, B@72, beta@104, C@112"

proc main =
  echo "[OK] vulkan push-constant block: tensor view = 32 bytes, GEMM block = 144 bytes"

when isMainModule:
  main()
