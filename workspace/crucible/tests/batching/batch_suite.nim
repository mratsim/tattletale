## Run from the worktree root (metal shown, same command per backend):
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

proc fillSentinel[E: HwEngine](engine: E, dst: auto, len: int, v: auto) =
  var host = newSeq[typeof(v)](len)
  for i in 0 ..< len:
    host[i] = v
  engine.copyToDeviceFromHost(dst, host.toOpenArray(0, len - 1))

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

  var tid32D = engine.allocDevice(uint32, TidN)
  var tid64D = engine.allocDevice(uint32, TidN)
  var sizes = engine.recordBatch()
  sizes.enqueue<<(2, 32)>> tidKernel(tid32D)
  sizes.enqueue<<(2, 64)>> tidKernel(tid64D)

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

    fillSentinel(engine, out2D, N, -1.0'f32)
    fillSentinel(engine, fillD, N, 0xFFFFFFFF'u32)
    multi.submit()
    var gotScaled: array[N, float32]
    var gotFilled: array[N, uint32]
    engine.copyToHostFromDevice(gotScaled.toOpenArray(0, N - 1), out2D)
    copyBack(engine, gotFilled, fillD)
    for i in 0 ..< N:
      doAssert gotScaled[i] == eagerScaled[i], "scaled mismatch at " & $i
      doAssert gotFilled[i] == eagerFilled[i], "filled mismatch at " & $i
    fillSentinel(engine, fillD, N, 0xFFFFFFFF'u32)
    multi.replay()
    copyBack(engine, gotFilled, fillD)
    for i in 0 ..< N:
      doAssert gotFilled[i] == 5'u32, "filled replay mismatch at " & $i
    echo "[OK] two different kernels: batch matches eager, replay matches"

  block:
    # same kernel at two workgroup sizes, each entry dispatches its own size
    engine.launchDeviceFn tidKernel(tid64D) do:
      grid = (2, 1)
      blk = (64, 1)
    var eager: array[TidN, uint32]
    copyBack(engine, eager, tid64D)
    for i in 0 ..< TidN:
      doAssert eager[i] == uint32(i mod 64), "eager tid mismatch at " & $i

    fillSentinel(engine, tid32D, TidN, 0xFFFFFFFF'u32)
    fillSentinel(engine, tid64D, TidN, 0xFFFFFFFF'u32)
    sizes.enqueue<<(2, 32)>> tidKernel(tid32D)
    sizes.enqueue<<(2, 64)>> tidKernel(tid64D)
    sizes.submit()
    var got32: array[TidN, uint32]
    var got64: array[TidN, uint32]
    copyBack(engine, got32, tid32D)
    copyBack(engine, got64, tid64D)
    for i in 0 ..< TidN:
      let expected32 = if i < 64: uint32(i mod 32) else: 0xFFFFFFFF'u32
      doAssert got32[i] == expected32, "batched 32-thread tid mismatch at " & $i
      doAssert got64[i] == uint32(i mod 64), "batched 64-thread tid mismatch at " & $i
    echo "[OK] two workgroup sizes: both entries dispatch their own size"

  block:
    # static argument baked into the wrapper, runtime argument packed
    engine.launchDeviceFn statKernel(statD, N, 7'u32) do:
      grid = (1, 1)
      blk = (16, 1)
    var eager: array[N, uint32]
    copyBack(engine, eager, statD)

    fillSentinel(engine, statD, N, 0xFFFFFFFF'u32)
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

    fillSentinel(engine, fillD, N, 0xFFFFFFFF'u32)
    fillSentinel(engine, productD, N, 0xFFFFFFFF'u32)
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

  block:
    # an intermediate buffer whose only owner left scope before submit: the batch retains what it records,
    # so both launches still see live memory
    var outD = engine.allocDevice(float32, N)
    var chained = engine.recordBatch()
    block:
      let tmp = engine.allocDevice(float32, N)
      chained.enqueue<<(1, 16)>> scaleKernel(tmp, xD, 3.0'f32)
      chained.enqueue<<(1, 16)>> scaleKernel(outD, tmp, 2.0'f32)
    doAssert chained.entries.len == 2
    fillSentinel(engine, outD, N, -1.0'f32)
    chained.submit()
    var got: array[N, float32]
    engine.copyToHostFromDevice(got.toOpenArray(0, N - 1), outD)
    for i in 0 ..< N:
      doAssert got[i] == 6.0'f32 * xs[i],
        "scope-freed intermediate mismatch at " & $i & ": " & $got[i]
    echo "[OK] scope-freed intermediate: the batch retains what it records"

  echo "[OK] batch suite (", engine.deviceName(), "): all scenarios passed"
