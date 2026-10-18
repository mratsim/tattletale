## NVRTC: 1D/2D/3D thread and block addressing, every thread records its linear
## global id and the raw blockIdx/threadIdx/blockDim components, the grid dims
## and block dims stay distinct per axis.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/device/t_nvrtc_thread_id.nim

import std/strutils
import workspace/crucible

const recordLen = 10 # per-thread record: gid, blockIdx.x/y/z, threadIdx.x/y/z, blockDim.x/y/z

proc checkRecord(rec: ptr UncheckedArray[uint32],
                 gid, bx, by, bz, tx, ty, tz, dx, dy, dz: int) =
  ## Assert every field of one thread's emitted record equals its exact expected
  ## value. Any mismatching field means the coordinate was not read through.
  let base = gid * recordLen
  doAssert rec[base + 0] == uint32(gid), "linear gid mismatch at gid " & $gid
  doAssert rec[base + 1] == uint32(bx), "blockIdx.x mismatch at gid " & $gid
  doAssert rec[base + 2] == uint32(by), "blockIdx.y mismatch at gid " & $gid
  doAssert rec[base + 3] == uint32(bz), "blockIdx.z mismatch at gid " & $gid
  doAssert rec[base + 4] == uint32(tx), "threadIdx.x mismatch at gid " & $gid
  doAssert rec[base + 5] == uint32(ty), "threadIdx.y mismatch at gid " & $gid
  doAssert rec[base + 6] == uint32(tz), "threadIdx.z mismatch at gid " & $gid
  doAssert rec[base + 7] == uint32(dx), "blockDim.x mismatch at gid " & $gid
  doAssert rec[base + 8] == uint32(dy), "blockDim.y mismatch at gid " & $gid
  doAssert rec[base + 9] == uint32(dz), "blockDim.z mismatch at gid " & $gid

# ---------------------------------------------------------------- 1D kernels
const kernelScalar = cuda:
  proc threadIdKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    output[0] = uint32(tid)
    output[1] = uint32(blockIdx.x)
    output[2] = uint32(threadIdx.x)
    output[3] = uint32(blockDim.x)

const kernel1d = cuda:
  proc threadState1d(rec: ptr UncheckedArray[uint32]) {.global.} =
    let gid = blockIdx.x * blockDim.x + threadIdx.x
    let base = gid * 10
    rec[base + 0] = uint32(gid)
    rec[base + 1] = uint32(blockIdx.x)
    rec[base + 2] = uint32(blockIdx.y)
    rec[base + 3] = uint32(blockIdx.z)
    rec[base + 4] = uint32(threadIdx.x)
    rec[base + 5] = uint32(threadIdx.y)
    rec[base + 6] = uint32(threadIdx.z)
    rec[base + 7] = uint32(blockDim.x)
    rec[base + 8] = uint32(blockDim.y)
    rec[base + 9] = uint32(blockDim.z)

# ----------------------------------------------------------------- 2D kernel
const kernel2d = cuda:
  proc threadState2d(rec: ptr UncheckedArray[uint32]) {.global.} =
    let blockId = blockIdx.y * gridDim.x + blockIdx.x
    let threadId = threadIdx.y * blockDim.x + threadIdx.x
    let gid = blockId * (blockDim.x * blockDim.y) + threadId
    let base = gid * 10
    rec[base + 0] = uint32(gid)
    rec[base + 1] = uint32(blockIdx.x)
    rec[base + 2] = uint32(blockIdx.y)
    rec[base + 3] = uint32(blockIdx.z)
    rec[base + 4] = uint32(threadIdx.x)
    rec[base + 5] = uint32(threadIdx.y)
    rec[base + 6] = uint32(threadIdx.z)
    rec[base + 7] = uint32(blockDim.x)
    rec[base + 8] = uint32(blockDim.y)
    rec[base + 9] = uint32(blockDim.z)

# ----------------------------------------------------------------- 3D kernel
const kernel3d = cuda:
  proc threadState3d(rec: ptr UncheckedArray[uint32]) {.global.} =
    let blockId = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x
    let threadId = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x
    let gid = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId
    let base = gid * 10
    rec[base + 0] = uint32(gid)
    rec[base + 1] = uint32(blockIdx.x)
    rec[base + 2] = uint32(blockIdx.y)
    rec[base + 3] = uint32(blockIdx.z)
    rec[base + 4] = uint32(threadIdx.x)
    rec[base + 5] = uint32(threadIdx.y)
    rec[base + 6] = uint32(threadIdx.z)
    rec[base + 7] = uint32(blockDim.x)
    rec[base + 8] = uint32(blockDim.y)
    rec[base + 9] = uint32(blockDim.z)
# ---------------------------------------------------------------------------

proc runTest() =   # private — engines are destroyed at proc return (F2 discipline)
  block: # 1D scalar: single block, single thread (field-read baseline)
    var engine = bkCuda.init(kernelScalar)
    echo "PTX: ", engine.getArtifact().len, " bytes"
    var buf: array[4, uint32]
    let o0 = engine.allocDevice(uint32, buf.len)
    engine.launchKernel<<(1, 1)>>("threadIdKernel", o0, ())
    engine.copyToHostFromDevice(addr buf[0], o0, buf.len)
    engine.freeDevice(o0)
    doAssert buf[0] == 0
    doAssert buf[1] == 0
    doAssert buf[2] == 0
    doAssert buf[3] == 1
    echo "  OK 1D scalar (1x1)"

  block: # 1D addressing: 3 blocks x 5 threads, distinct linear gids (asymmetric
         # grid vs block so an exchange of the two extents is caught)
    var engine = bkCuda.init(kernel1d)
    var rec: array[3 * 5 * recordLen, uint32]
    let o1 = engine.allocDevice(uint32, rec.len)
    engine.launchKernel<<(3, 5)>>("threadState1d", o1, ())
    engine.copyToHostFromDevice(addr rec[0], o1, rec.len)
    engine.freeDevice(o1)
    for b in 0 ..< 3:
      for t in 0 ..< 5:
        let gid = b * 5 + t
        checkRecord(cast[ptr UncheckedArray[uint32]](addr rec),
                    gid, b, 0, 0, t, 0, 0, 5, 1, 1)
    echo "  OK 1D addressing (grid 3, block 5)"

  block: # 2D addressing: grid (2,3), block (4,2) -- uses blockIdx.y/blockDim.y
    const gx = 2
    const gy = 3
    const bx = 4
    const by = 2
    var engine = bkCuda.init(kernel2d)
    var rec: array[gx * gy * bx * by * recordLen, uint32]
    let o2 = engine.allocDevice(uint32, rec.len)
    engine.launchKernel<<(grid: (gx, gy), blk: (bx, by))>>("threadState2d", o2, ())
    engine.copyToHostFromDevice(addr rec[0], o2, rec.len)
    engine.freeDevice(o2)
    for byy in 0 ..< gy:
      for bxx in 0 ..< gx:
        let blockId = byy * gx + bxx
        for tyy in 0 ..< by:
          for txx in 0 ..< bx:
            let threadId = tyy * bx + txx
            let gid = blockId * (bx * by) + threadId
            checkRecord(cast[ptr UncheckedArray[uint32]](addr rec),
                        gid, bxx, byy, 0, txx, tyy, 0, bx, by, 1)
    echo "  OK 2D addressing (grid 2x3, block 4x2)"

  block: # 3D addressing: grid (2,3,4), block (4,2,3) -- every axis > 1 and
         # mutually distinct (gy != gz, by != bz, grid.x != block.x) so a swap of
         # any two axes is caught
    const gx = 2
    const gy = 3
    const gz = 4
    const bx = 4
    const by = 2
    const bz = 3
    var engine = bkCuda.init(kernel3d)
    var rec: array[gx * gy * gz * bx * by * bz * recordLen, uint32]
    let o3 = engine.allocDevice(uint32, rec.len)
    engine.launchKernel<<(grid: (gx, gy, gz), blk: (bx, by, bz))>>("threadState3d", o3, ())
    engine.copyToHostFromDevice(addr rec[0], o3, rec.len)
    engine.freeDevice(o3)
    for bzz in 0 ..< gz:
      for byy in 0 ..< gy:
        for bxx in 0 ..< gx:
          let blockId = (bzz * gy + byy) * gx + bxx
          for tzz in 0 ..< bz:
            for tyy in 0 ..< by:
              for txx in 0 ..< bx:
                let threadId = (tzz * by + tyy) * bx + txx
                let gid = blockId * (bx * by * bz) + threadId
                checkRecord(cast[ptr UncheckedArray[uint32]](addr rec),
                            gid, bxx, byy, bzz, txx, tyy, tzz, bx, by, bz)
    echo "  OK 3D addressing (grid 2x3x4, block 4x2x3)"

when isMainModule:
  runTest()
  echo repeat('-', 60)
  echo "OK (test_nvrtc_thread_id): 1D, 2D and 3D addressing verified"
