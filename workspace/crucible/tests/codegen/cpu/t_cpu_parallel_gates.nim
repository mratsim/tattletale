## CPU engine: parallel determinism checks over the threadpool_compute pool.
##
## Test that the pooled launch reproduces the serial launch output bitwise.
## Test that axpy and workgroup-tag outputs stay identical at pool sizes 1, 4, 20 in both thread modes.
## Test that K = 1, 4, grid size coarsening stays bitwise identical across pools in pooled mode (the serial launch ignores the knob).
##
## Run:
##   nim test_crucible_cpu
import std/strformat
import workspace/crucible

proc axpyPar(y: ptr UncheckedArray[float32], alpha: float32, x: ptr UncheckedArray[float32], n: int32) =
  let gid = int(thread_position_in_grid.x)
  if gid < n:
    y[gid] = alpha * x[gid] + y[gid]

proc tagWorkgroups(outBuf: ptr UncheckedArray[uint32], n: int32) =
  let wg = threadgroup_position_in_grid.x
  let gid = int(thread_position_in_grid.x)
  if gid < n:
    outBuf[gid] = uint32(wg)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # Odd sizes stress the workgroup decomposition: no pool count or block size divides N.
  const N = 257
  const BLK = 8

  proc hostArrays(): tuple[x: array[N, float32], y: array[N, float32], expected: array[N, float32]] =
    for i in 0 ..< N:
      result.x[i] = float32(i) + 0.5'f32
      result.y[i] = float32(10 * (i + 1))
      result.expected[i] = 2.5'f32 * result.x[i] + result.y[i]

  proc launchAxpy(engine: CpuEngine, y, x: ptr UncheckedArray[float32]) =
    engine.launchCpuFn axpyPar(y, 2.5'f32, x, int32(N)) do:
      grid = ((N + BLK - 1) div BLK, 1)
      blk = (BLK, 1)

  block:   # bitwise check, axpy over float32
    let (xh, yh0, expected) = hostArrays()
    var refOut: array[N, float32]
    for pool in [1, 4, 20]:
      var xh = xh
      var yh = yh0
      var engine = newCpuEngine(pool)
      launchAxpy(engine, cast[ptr UncheckedArray[float32]](addr yh[0]),
        cast[ptr UncheckedArray[float32]](addr xh[0]))
      engine.destroyCpuEngine()
      for i in 0 ..< N:
        doAssert yh[i] == expected[i],
          &"axpy pool {pool} [{i}]: got {yh[i]} want {expected[i]}"
      if pool == 1:
        refOut = yh
      else:
        for i in 0 ..< N:
          doAssert yh[i] == refOut[i],
            &"bitwise gate pool {pool} [{i}]: got {yh[i]} want {refOut[i]}"
    echo "[OK] bitwise check: axpy identical across pools 1, 4, 20"

  block:   # injectivity check, one workgroup id per cell
    let wgCount = (N + BLK - 1) div BLK
    for pool in [1, 4, 20]:
      var buf: array[N, uint32]
      var engine = newCpuEngine(pool)
      engine.launchCpuFn tagWorkgroups(cast[ptr UncheckedArray[uint32]](addr buf[0]), int32(N)) do:
        grid = (wgCount, 1)
        blk = (BLK, 1)
      engine.destroyCpuEngine()
      for gid in 0 ..< N:
        let want = uint32(gid div BLK)
        doAssert buf[gid] == want,
          &"injectivity pool {pool} [{gid}]: got {buf[gid]} want {want}"
    echo "[OK] injectivity check: workgroup tags exact across pools 1, 4, 20"

  block:   # coarsening check, K adjacent workgroups share one pool task
    # 33 workgroups: K = 4 leaves a one-workgroup tail task, K = 33 runs
    # all workgroups in one task
    let wgCount = (N + BLK - 1) div BLK
    let (xh, yh0, expected) = hostArrays()
    var refOut: array[N, float32]
    when compileOption("threads"):
      for k in [1, 4, wgCount]:
        for pool in [1, 4, 20]:
          var xh = xh
          var yh = yh0
          var buf: array[N, uint32]
          var engine = newCpuEngine(pool)
          engine.launchCpuFn axpyPar(cast[ptr UncheckedArray[float32]](addr yh[0]),
              2.5'f32,
              cast[ptr UncheckedArray[float32]](addr xh[0]), int32(N)) do:
            grid = (wgCount, 1)
            blk = (BLK, 1)
            workgroupsPerTask = k
          engine.launchCpuFn tagWorkgroups(cast[ptr UncheckedArray[uint32]](addr buf[0]),
              int32(N)) do:
            grid = (wgCount, 1)
            blk = (BLK, 1)
            workgroupsPerTask = k
          engine.destroyCpuEngine()
          for i in 0 ..< N:
            doAssert yh[i] == expected[i],
              &"coarse axpy K={k} pool {pool} [{i}]: got {yh[i]} want {expected[i]}"
            doAssert buf[i] == uint32(i div BLK),
              &"coarse tag K={k} pool {pool} [{i}]: got {buf[i]} want {uint32(i div BLK)}"
          if k == 1 and pool == 1:
            refOut = yh
          else:
            for i in 0 ..< N:
              doAssert yh[i] == refOut[i],
                &"coarse bitwise K={k} pool {pool} [{i}]: got {yh[i]} want {refOut[i]}"
      echo "[OK] coarsening check: axpy and workgroup tags bitwise identical at K = 1, 4, grid across pools 1, 4, 20"
    else:
      # serial mode covers the whole grid in one call, a literal knob
      # is accepted and ignored
      var yh = yh0
      var buf: array[N, uint32]
      var engine = newCpuEngine(1)
      engine.launchCpuFn axpyPar(cast[ptr UncheckedArray[float32]](addr yh[0]),
          2.5'f32,
          cast[ptr UncheckedArray[float32]](addr xh[0]), int32(N)) do:
        grid = (wgCount, 1)
        blk = (BLK, 1)
        workgroupsPerTask = 4
      engine.launchCpuFn tagWorkgroups(cast[ptr UncheckedArray[uint32]](addr buf[0]),
          int32(N)) do:
        grid = (wgCount, 1)
        blk = (BLK, 1)
        workgroupsPerTask = 4
      engine.destroyCpuEngine()
      for i in 0 ..< N:
        doAssert yh[i] == expected[i],
          &"serial axpy [{i}]: got {yh[i]} want {expected[i]}"
        doAssert buf[i] == uint32(i div BLK),
          &"serial tag [{i}]: got {buf[i]} want {uint32(i div BLK)}"
      echo "[OK] coarsening check: the serial launch ignores the knob and covers the whole grid"

runTest()
echo "t_cpu_parallel_gates: all checks passed"
