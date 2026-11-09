## Engine-agnostic batching suite: every scenario runs through the batch
## API (`recordBatch`/`enqueue`/`submit`/`replay`) and is judged against an eager
## `launchDeviceFn` launch on the same engine.
##
## One driver file per runtime instantiates the suite:
##   t_batch_metal.nim, t_batch_vulkan.nim, t_batch_opencl.nim, t_batch_webgpu.nim
##
## Enqueue order is compile-time order:
## - every batch records its entries before any `submit` runs
## - each batch then submits and checks
## - compile streams share their CIR procs between finalizes, a finalize over new enqueues re-runs the lowering passes and fails
## - one compile per compilation, later finalizes reuse the artifact
##
## Every scenario runs on every backend, no skips. A known failure stands
## as a finding, never as a condition:
## - WebGPU compiles exactly one global function per module.
##   The multi-kernel scenarios fail the whole-module compile.
##
## Run from the worktree root (metal shown, same command per runtime):
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_batch_metal.nim

import workspace/crucible

const
  N = 16
  TidN = 128

proc scaleKernel*(outP, x: ptr UncheckedArray[float32], s: float32) =
  let i = int(thread_position_in_grid.x)
  outP[i] = s * x[i]

proc fillKernel*(outP: ptr UncheckedArray[uint32], v: uint32) =
  let i = int(thread_position_in_grid.x)
  if i < N:
    outP[i] = v

proc fillKernel*(outP: ptr UncheckedArray[uint32], v, scale: uint32) =
  let i = int(thread_position_in_grid.x)
  if i < N:
    outP[i] = v * scale

proc tidKernel*(outP: ptr UncheckedArray[uint32]) =
  # thread_position_in_threadgroup reads back the running workgroup size, each entry's own size is observable in the output.
  let i = int(thread_position_in_grid.x)
  let t = uint32(thread_position_in_threadgroup.x)
  if i < TidN:
    outP[i] = t

proc statKernel*(outP: ptr UncheckedArray[uint32], n: static int, v: uint32) =
  let i = int(thread_position_in_grid.x)
  if i < n:
    outP[i] = v + uint32(n)

proc copyBack[E: HwEngine](engine: E, dst: var openArray[uint32], src: auto) =
  var host = newSeq[uint32](dst.len)
  engine.copyToHostFromDevice(host.toOpenArray(0, host.len - 1), src)
  for i in 0 ..< dst.len:
    dst[i] = host[i]

proc runBatchSuite*[E: HwEngine](engine: E) =
  var xs: array[N, float32]
  for i in 0 ..< N:
    xs[i] = float32(i)

  var xD = engine.allocDevice(float32, N)
  var out1D = engine.allocDevice(float32, N)
  engine.copyToDeviceFromHost(xD, xs.toOpenArray(0, N - 1))

  var single = engine.recordBatch()
  single.enqueue<<(1, 16)>> scaleKernel(out1D, xD, 3.0'f32)

  var out2D = engine.allocDevice(float32, N)
  var fillD = engine.allocDevice(uint32, N)
  var tidD = engine.allocDevice(uint32, TidN)
  var statD = engine.allocDevice(uint32, N)
  var productD = engine.allocDevice(uint32, N)

  var multi = engine.recordBatch()
  multi.enqueue<<(1, 16)>> scaleKernel(out2D, xD, 5.0'f32)
  multi.enqueue<<(1, 16)>> fillKernel(fillD, 5'u32)

  var sizes = engine.recordBatch()
  sizes.enqueue<<(2, 32)>> tidKernel(tidD)
  sizes.enqueue<<(2, 64)>> tidKernel(tidD)

  var statics = engine.recordBatch()
  statics.enqueue<<(1, 16)>> statKernel(statD, N, 7'u32)

  var overloads = engine.recordBatch()
  overloads.enqueue<<(1, 16)>> fillKernel(fillD, 5'u32)
  overloads.enqueue<<(1, 16)>> fillKernel(productD, 3'u32, 4'u32)

  # ── Submit phase: each batch launches its entries, judged against eager ──


  # ── Submit phase: each batch launches its entries, judged against eager ──

  # single kernel through submit, then two contents-only replays
  engine.launchDeviceFn scaleKernel(out1D, xD, 2.0'f32) do:
    grid = (1, 1)
    blk = (16, 1)
  var eager: array[N, float32]
  engine.copyToHostFromDevice(eager.toOpenArray(0, N - 1), out1D)

  single.submit()
  var got: array[N, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, N - 1), out1D)
  for i in 0 ..< N:
    doAssert got[i] == 3.0'f32 * xs[i], "submit mismatch at " & $i & ": " & $got[i]

  for r in 0 ..< 2:
    single.replay()
    engine.copyToHostFromDevice(got.toOpenArray(0, N - 1), out1D)
    for i in 0 ..< N:
      doAssert got[i] == 3.0'f32 * xs[i], "replay " & $r & " mismatch at " & $i

  # eager launches still run after the batch ingested its artifact
  engine.launchDeviceFn scaleKernel(out1D, xD, 2.0'f32) do:
    grid = (1, 1)
    blk = (16, 1)
  engine.copyToHostFromDevice(got.toOpenArray(0, N - 1), out1D)
  for i in 0 ..< N:
    doAssert got[i] == eager[i], "eager-after-batch mismatch at " & $i
  echo "[OK] single kernel: submit, two replays, eager interleaved"

  block:
    # two different kernels in one batch, each entry its own entrypoint
    engine.launchDeviceFn scaleKernel(out2D, xD, 5.0'f32) do:
      grid = (1, 1)
      blk = (16, 1)
    engine.launchDeviceFn fillKernel(fillD, 5'u32) do:
      grid = (1, 1)
      blk = (16, 1)
    var eagerScaled: array[N, float32]
    var eagerFilled: array[N, uint32]
    engine.copyToHostFromDevice(eagerScaled.toOpenArray(0, N - 1), out2D)
    copyBack(engine, eagerFilled, fillD)

    multi.submit()
    var gotScaled: array[N, float32]
    var gotFilled: array[N, uint32]
    engine.copyToHostFromDevice(gotScaled.toOpenArray(0, N - 1), out2D)
    copyBack(engine, gotFilled, fillD)
    for i in 0 ..< N:
      doAssert gotScaled[i] == eagerScaled[i], "scaled mismatch at " & $i
      doAssert gotFilled[i] == eagerFilled[i], "filled mismatch at " & $i
    multi.replay()
    copyBack(engine, gotFilled, fillD)
    for i in 0 ..< N:
      doAssert gotFilled[i] == 5'u32, "filled replay mismatch at " & $i
    echo "[OK] two different kernels: batch matches eager, replay matches"

  block:
    # same kernel at two workgroup sizes, each entry dispatches its own size
    engine.launchDeviceFn tidKernel(tidD) do:
      grid = (2, 1)
      blk = (64, 1)
    var eager: array[TidN, uint32]
    copyBack(engine, eager, tidD)
    for i in 0 ..< TidN:
      doAssert eager[i] == uint32(i mod 64), "eager tid mismatch at " & $i

    sizes.submit()
    var got: array[TidN, uint32]
    copyBack(engine, got, tidD)
    for i in 0 ..< TidN:
      doAssert got[i] == uint32(i mod 64), "batched tid mismatch at " & $i
    echo "[OK] two workgroup sizes: both entries dispatch their own size"

  block:
    # static argument baked into the wrapper, runtime argument packed
    engine.launchDeviceFn statKernel(statD, N, 7'u32) do:
      grid = (1, 1)
      blk = (16, 1)
    var eager: array[N, uint32]
    copyBack(engine, eager, statD)

    statics.submit()
    var got: array[N, uint32]
    copyBack(engine, got, statD)
    for i in 0 ..< N:
      doAssert got[i] == eager[i], "static-args mismatch at " & $i
      doAssert got[i] == uint32(N + 7), "static-args value mismatch at " & $i
    echo "[OK] static args: baked value reaches the kernel through the batch"

  block:
    # two overloads of one kernel, each entry dispatches on its argument count
    engine.launchDeviceFn fillKernel(fillD, 5'u32) do:
      grid = (1, 1)
      blk = (16, 1)
    engine.launchDeviceFn fillKernel(productD, 3'u32, 4'u32) do:
      grid = (1, 1)
      blk = (16, 1)
    var eagerFilled: array[N, uint32]
    var eagerProduct: array[N, uint32]
    copyBack(engine, eagerFilled, fillD)
    copyBack(engine, eagerProduct, productD)

    overloads.submit()
    var gotFilled: array[N, uint32]
    var gotProduct: array[N, uint32]
    copyBack(engine, gotFilled, fillD)
    copyBack(engine, gotProduct, productD)
    for i in 0 ..< N:
      doAssert gotFilled[i] == eagerFilled[i] and gotFilled[i] == 5'u32,
        "1-arg overload mismatch at " & $i
      doAssert gotProduct[i] == eagerProduct[i] and gotProduct[i] == 12'u32,
        "2-arg overload mismatch at " & $i
    echo "[OK] overloads: both entries ran their own overload"


  echo "[OK] batch suite (", engine.deviceName(), "): all scenarios passed"
