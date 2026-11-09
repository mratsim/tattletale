## Metal: a pointer cast wrapping a statement expression reads threadgroup storage
##
## Test that a cast whose operand is a statement expression derives the address
## space of the value the statement expression computes:
##
## - threadgroup call site through a cast over a block expression
## - launch + byte-exact readback
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_space_stmt_cast.nim

import workspace/crucible

proc readFirst(p: ptr UncheckedArray[float32]): float32 {.device.} =
  p[0]

const stmtCastSpaceMsl = metal:
  proc kern(outp: ptr UncheckedArray[float32]) {.global.} =
    var scratch {.smem.}: array[4, float32]
    let t = thread_position_in_threadgroup.x
    if t == 0:
      scratch[0] = 1.5'f32
    threadgroup_barrier()
    if t == 0:
      let q = addr scratch[0]
      outp[0] = readFirst(cast[ptr UncheckedArray[float32]](
        (block:
          let alias = q
          alias)))

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(stmtCastSpaceMsl)
  let dev = engine.allocDevice(float32, 1)
  engine.launchKernel<<(grid: (1, 1), blk: (4, 1))>>("kern", dev, ())
  var got: array[1, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 1 - 1), dev)
  doAssert got[0] == 1.5'f32,
    "kern readback: got " & $got[0] & ", want 1.5"
  echo "[OK] Metal: a cast over a statement expression adopts the value's address space"

when isMainModule:
  runTest()
