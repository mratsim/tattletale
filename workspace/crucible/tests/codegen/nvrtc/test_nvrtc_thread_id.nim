## NVRTC: 1D/2D/3D thread & block addressing via the codegen pipeline
##
## For 1D, 2D and 3D this test launches a grid/block of the matching rank on the
## GPU through the real NVRTC compile + execute path, then verifies that every
## thread wrote its exact linear global id -- computed from blockIdx.x/y/z,
## threadIdx.x/y/z and blockDim.x/y/z -- plus the raw value of each coordinate
## component. The 2D and 3D cases use y/z extents > 1 so that a coordinate that
## is ignored, collapsed or defaulted changes the emitted buffer and fails a
## doAssert.
##
## Extents are chosen with a distinct value on every distinguishing axis (grid vs
## block along x; y vs z within both grid and block), so that an axis swap in the
## launch plumbing -- e.g. grid and block exchanged, or blockIdx.y/z transposed --
## also changes the emitted buffer and fails a doAssert.
##
## Run (from tattletale/, CUDA 12 available):
##   export CUDA_HOME=/usr/local/cuda-12
##   export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64
##   nim cpp -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_thread_id.nim
##
import std/strutils
import workspace/crucible
# The legacy NVRTC driver (initNvrtc/execute) is not re-exported by engines.nim
# anymore (clean engine API only) — import it directly; order matters:
# engines must be processed first so its `import ./engines/nvrtc {.all.}`
# sees a fully-processed nvrtc module (the engines ↔ nvrtc circular import
# only compiles in that direction); the direct import below is then cached.
import workspace/crucible/src/runtime/engines/nvrtc
# TODO(engine): this test exercises 2D/3D launch extents (dim3 grid/block) which
# are not expressible via the 1D engine LaunchConfig — kept on the internal
# NVRTC execute path on purpose.
import workspace/crucible/src/runtime/exec/cuda_runtime

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

block: # 1D scalar: single block, single thread (field-read baseline)
  var nv = initNvrtc(kernelScalar)
  nv.numBlocks = 1
  nv.threadsPerBlock = 1
  nv.compile()
  nv.getPtx()
  echo "PTX: ", nv.ptx.len, " bytes"
  var buf: array[4, uint32]
  nv.execute("threadIdKernel", buf, ())
  doAssert buf[0] == 0
  doAssert buf[1] == 0
  doAssert buf[2] == 0
  doAssert buf[3] == 1
  echo "  OK 1D scalar (1x1)"

block: # 1D addressing: 3 blocks x 5 threads, distinct linear gids (asymmetric
       # grid vs block so an exchange of the two extents is caught)
  var nv = initNvrtc(kernel1d)
  nv.numBlocks = 3
  nv.threadsPerBlock = 5
  nv.compile()
  nv.getPtx()
  var rec: array[3 * 5 * recordLen, uint32]
  nv.execute("threadState1d", rec, ())
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
  var nv = initNvrtc(kernel2d)
  nv.compile()
  nv.getPtx()
  var rec: array[gx * gy * bx * by * recordLen, uint32]
  nv.execute("threadState2d", dim3(gx, gy), dim3(bx, by), rec, ())
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
  var nv = initNvrtc(kernel3d)
  nv.compile()
  nv.getPtx()
  var rec: array[gx * gy * gz * bx * by * bz * recordLen, uint32]
  nv.execute("threadState3d", dim3(gx, gy, gz), dim3(bx, by, bz), rec, ())
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

echo repeat('-', 60)
echo "OK (test_nvrtc_thread_id): 1D, 2D and 3D addressing verified"