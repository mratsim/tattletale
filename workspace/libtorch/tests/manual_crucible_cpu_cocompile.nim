# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU engine + libtorch in one nim cpp build (co-compile test).
##
## Test that a crucible launchCpuFn kernel and a libtorch Tensor co-compile into one
## nim cpp binary and compute the same axpy.
##
## Run:
##   nim cpp -r --threads:on -d:release --passC:"-std=c++20" \
##     --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/libtorch/tests/manual_crucible_cpu_cocompile.nim
import workspace/crucible
import workspace/libtorch
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed

type
  AxpyView = object
    p: ptr UncheckedArray[float32]
    n: int

proc axpy(y: AxpyView, alpha: float32, x: AxpyView) =
  for i in 0 ..< y.n:
    y.p[i] = alpha * x.p[i] + y.p[i]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  const N = 7
  manual_seed(Torch, 42)
  let xT = rand(N, kFloat32)
  let yT = rand(N, kFloat32)
  let xSrc = xT.contiguous().data_ptr(float32)
  let ySrc = yT.contiguous().data_ptr(float32)
  var xHost: array[N, float32]
  var yHost: array[N, float32]
  for i in 0 ..< N:
    xHost[i] = xSrc[i]
    yHost[i] = ySrc[i]

  engine.launchCpuFn axpy(AxpyView(p: cast[ptr UncheckedArray[float32]](addr yHost[0]), n: N),
                          2.5'f32,
                          AxpyView(p: cast[ptr UncheckedArray[float32]](addr xHost[0]), n: N)) do:
    grid = (1, 1)
    blk = (1, 1)

  runCppTest "libtorch reference agrees with the crucible axpy kernel":
    proc(): bool =
      let refT = xT * 2.5'f32 + yT
      let refPtr = refT.contiguous().data_ptr(float32)
      for i in 0 ..< N:
        # Kernel alpha*x+y may fuse into an FMA on arm64 while libtorch
        # computes a multiply-then-add reference, results one float32 ulp apart
        doAssert abs(refPtr[i] - yHost[i]) <= abs(refPtr[i]) * 1e-6'f32 + 1e-6'f32,
          "element " & $i & ": kernel " & $yHost[i] & " vs reference " & $refPtr[i]
      true

  echo "[OK] co-compile: crucible launchCpuFn and libtorch in one nim cpp binary"

runTest()
