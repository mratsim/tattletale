# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CUDA Hello World test for Positron.
##
## Verifies basic CUDA device init, memory allocation, and a trivial kernel.
## Compiled with `--cc:nvcc` via `test_cuda_hello.nim.cfg`.
##
## Run from tattletale root:
##   CUDA_HOME=".venv/lib/python3.14/site-packages/nvidia/cu13" \
##   LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/../torch/lib" \
##   PATH="$CUDA_HOME/bin:$PATH" \
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests/positron --nimcache:nimcache/tests/positron \
##     workspace/positron/tests/test_cuda_hello.nim

import
  std/math,
  workspace/positron/experimental/platforms/cuda,
  workspace/positron/experimental/nimcuda/rms_norm_cuda

# ############################################################
#
#               Trivial vec-add kernel
#
# ############################################################

proc vecAddKernel(a: ptr UncheckedArray[cfloat],
                  b: ptr UncheckedArray[cfloat],
                  c: ptr UncheckedArray[cfloat],
                  n: cint) {.
  exportc, codegenDecl: "__global__ $# $#$#".} =
  ## Simple vector add: c[i] = a[i] + b[i]
  let i = blockIdx.x.cint * blockDim.x.cint + threadIdx.x.cint
  if i < n:
    c[i] = a[i] + b[i]

proc testVecAdd() =
  ## Test: allocate two float arrays on GPU, add them, verify result.

  const N = 256
  let nBytes = (N * sizeof(cfloat)).csize_t

  # Host data
  var a_h, b_h, c_h: array[N, cfloat]
  for i in 0..<N:
    a_h[i] = cfloat(i)
    b_h[i] = cfloat(i * 2)

  # Device allocations
  var a_d, b_d, c_d: pointer
  check cudaMalloc(addr a_d, nBytes)
  check cudaMalloc(addr b_d, nBytes)
  check cudaMalloc(addr c_d, nBytes)

  # Copy host → device
  check cudaMemcpy(a_d, addr a_h[0], nBytes, cudaMemcpyHostToDevice)
  check cudaMemcpy(b_d, addr b_h[0], nBytes, cudaMemcpyHostToDevice)

  # Launch kernel
  let gridSize = (N + 255) div 256
  let a_dev = cast[ptr UncheckedArray[cfloat]](a_d)
  let b_dev = cast[ptr UncheckedArray[cfloat]](b_d)
  let c_dev = cast[ptr UncheckedArray[cfloat]](c_d)

  var args: array[4, pointer]
  args[0] = cast[pointer](addr a_dev)
  args[1] = cast[pointer](addr b_dev)
  args[2] = cast[pointer](addr c_dev)
  let nVal = N.cint
  args[3] = cast[pointer](addr nVal)

  let gridDim = dim3(gridSize)
  let blockDim = dim3(256)

  check cudaLaunchKernel(
    vecAddKernel,
    gridDim, blockDim,
    args[0].addr,
    0.csize_t, nil
  )

  check cudaDeviceSynchronize()

  # Copy device → host
  check cudaMemcpy(addr c_h[0], c_d, nBytes, cudaMemcpyDeviceToHost)

  # Verify
  var allOk = true
  for i in 0..<N:
    let expected = cfloat(i) + cfloat(i * 2)
    if abs(c_h[i] - expected) > 1e-6:
      stderr.writeLine("Mismatch at [", i, "]: got ", c_h[i], " expected ", expected)
      allOk = false

  # Cleanup
  check cudaFree(a_d)
  check cudaFree(b_d)
  check cudaFree(c_d)

  if allOk:
    echo "✅ testVecAdd: PASS (", N, " elements)"
  else:
    stderr.writeLine("❌ testVecAdd: FAIL")
    quit(1)

# ############################################################
#
#                     RMSNorm test
#
# ############################################################

proc testRmsNorm() =
  ## Test RMSNorm kernel against a manual fp32 computation.

  let rows = 2.cint
  let dim = 256.cint  # divisible by 4
  let eps = 1e-6.cfloat
  let nElements = (rows * dim).int
  let nBytes = (nElements * sizeof(Half)).csize_t

  # Host data
  var x_h = newSeq[Half](nElements)
  var w_h = newSeq[Half](dim)
  var y_h = newSeq[Half](nElements)

  # Fill with deterministic values (fp16 bit patterns)
  for i in 0..<nElements:
    x_h[i] = float2halfRn(cfloat(sin(float(i)) * 10.0))
  for i in 0..<dim:
    w_h[i] = float2halfRn(cfloat(cos(float(i * 3)) * 2.0 + 1.0))

  # Reference: compute in fp32 via manual loop
  var y_ref = newSeq[cfloat](nElements)
  for row in 0..<rows:
    var ss: cfloat = 0.0
    for col in 0..<dim.int:
      let xf = half2float(x_h[row * dim.int + col])
      ss += xf * xf
    let rmf = rsqrtf(ss / cfloat(dim) + eps)
    for col in 0..<dim.int:
      let xf = half2float(x_h[row * dim.int + col])
      let wf = half2float(w_h[col])
      y_ref[row * dim.int + col] = xf * wf * rmf

  # Device allocations
  var x_d, w_d, y_d: pointer
  check cudaMalloc(addr x_d, nBytes)
  check cudaMalloc(addr w_d, csize_t(dim * sizeof(Half)))
  check cudaMalloc(addr y_d, nBytes)

  # Copy host → device
  check cudaMemcpy(x_d, addr x_h[0], nBytes, cudaMemcpyHostToDevice)
  check cudaMemcpy(w_d, addr w_h[0], csize_t(dim * sizeof(Half)), cudaMemcpyHostToDevice)

  # Launch RMSNorm kernel
  let x_dev = cast[pHalf](x_d)
  let w_dev = cast[pHalf](w_d)
  let y_dev = cast[pHalf](y_d)

  rmsNormCuda_fp16(x_dev, w_dev, y_dev, eps, rows, dim)

  # Copy result back
  check cudaMemcpy(addr y_h[0], y_d, nBytes, cudaMemcpyDeviceToHost)

  # Verify
  var maxDiff: cfloat = 0.0
  for i in 0..<nElements:
    let computed = half2float(y_h[i])
    let expected = y_ref[i]
    let diff = abs(computed - expected)
    if diff > maxDiff:
      maxDiff = diff

  # For fp16, tolerance of ~0.1 is reasonable for functional correctness
  let tol = 0.1'f32
  if maxDiff <= tol:
    echo "✅ testRmsNorm: PASS (max_diff=", maxDiff, ", tol=", tol, ")"
  else:
    stderr.writeLine("❌ testRmsNorm: FAIL (max_diff=", maxDiff, " > tol=", tol, ")")
    for i in 0..<min(8, nElements):
      stderr.writeLine("  [", i, "] computed=", half2float(y_h[i]), " expected=", y_ref[i])
    quit(1)

  # Cleanup
  check cudaFree(x_d)
  check cudaFree(w_d)
  check cudaFree(y_d)

# ############################################################
#
#                         Main
#
# ############################################################

when isMainModule:
  echo "Positron CUDA Hello World"
  echo "═════════════════════════"
  echo ""

  testVecAdd()
  echo ""
  testRmsNorm()
  echo ""
  echo "All tests passed."
