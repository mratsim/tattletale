## Engine-agnostic batching suite: every scenario runs through the tier-2 batch
## API (`recordBatch`/`enqueue`/`submit`/`replay`) and is judged against an eager
## `launchDeviceFn` launch on the same engine.
##
## One driver file per runtime instantiates the suite:
##   t_batch_metal.nim, t_batch_vulkan.nim, t_batch_opencl.nim, t_batch_webgpu.nim
##
## Enqueue order is compile-time order:
## - every batch records its entries before any `submit` runs
## - each batch then submits and checks
## - tier-2 compile streams share their CIR procs between finalizes, a finalize over new enqueues re-runs the lowering passes and fails
## - one compile per compilation, later finalizes reuse the artifact
##
## Backend capability conditions, compile-time `when`s with a printed note:
## - threadgroup memory is Metal-only: the Vulkan, OpenCL and WebGPU tier-1
##   batches reject launch-time sharedMem.
## - WebGPU compiles exactly one global function per module, so its instance
##   runs the single-kernel scenario only. A second distinct kernel fails the whole-module compile.
## - the OpenCL tier-1 record path packs the engine's OpenCLBuffer handle as scalar bytes
##   while the eager launch path binds it through a dedicated store overload, so its instance
##   runs no scenario until handles bind.
##
## Run from the worktree root (metal shown, same command per runtime):
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_batch_metal.nim

import workspace/crucible

const
  N = 16
  TidN = 128
  Blk = 64
  Groups = 4
  Sum = float32(Blk * (Blk - 1) div 2)  # 0 + 1 + ... + 63

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

proc tgSumKernel*(output: ptr UncheckedArray[float32]) =
  var scratch {.smem.}: array[Blk, float32]
  let t = int(thread_position_in_threadgroup.x)
  scratch[t] = float32(t)
  syncthreads()
  if t == 0:
    var s = 0.0'f32
    for i in 0 ..< Blk:
      s += scratch[i]
    output[int(threadgroup_position_in_grid.x)] = s

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

  when E is not WebgpuEngine and E is not OpenCLEngine:
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

  # ── Metal scenario: its submit is the compilation's first finalize ──────

  when E is MetalEngine:
    block:
      # threadgroup memory encoded per entry, submit and replay
      var tgD = engine.allocDevice(float32, Groups)
      var smem = engine.recordBatch()
      smem.enqueue<<(Groups, Blk, Blk * sizeof(float32))>> tgSumKernel(tgD)

      engine.launchDeviceFn tgSumKernel(tgD) do:
        grid = (Groups, 1)
        blk = (Blk, 1)
        sharedMem = Blk * sizeof(float32)
      var eager: array[Groups, float32]
      engine.copyToHostFromDevice(eager.toOpenArray(0, Groups - 1), tgD)
      for g in 0 ..< Groups:
        doAssert eager[g] == Sum, "eager smem mismatch at " & $g

      smem.submit()
      var got: array[Groups, float32]
      engine.copyToHostFromDevice(got.toOpenArray(0, Groups - 1), tgD)
      for g in 0 ..< Groups:
        doAssert got[g] == Sum, "batched smem mismatch at " & $g
      smem.replay()
      engine.copyToHostFromDevice(got.toOpenArray(0, Groups - 1), tgD)
      for g in 0 ..< Groups:
        doAssert got[g] == Sum, "smem replay mismatch at " & $g
      echo "[OK] threadgroup memory: batch and replay match eager"

  # ── Submit phase: each batch launches its entries, judged against eager ──

  when E is not OpenCLEngine:
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

  when E is not WebgpuEngine and E is not OpenCLEngine:
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

  when E is OpenCLEngine:
    echo "[BLOCKED] OpenCL: the leg's record path packs OpenCLBuffer handles as " &
      "scalar bytes, every scenario fails at the first kernel-arg bind (see " &
      "TRACEABILITY.md, finding F5)"

  echo "[OK] batch suite (", engine.deviceName(), "): all scenarios passed"
