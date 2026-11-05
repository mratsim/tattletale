# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedding gather backward kernel: scatter-add of the cotangent into the rank-2 embedding table gradient.
##
## With out[r, c] = weight[idx[r], c] as the gather forward, g the cotangent of the gather output
## and dw the zero-initialized table gradient (vocabulary, e):
##   dw[idx[r], c] += g[r, c]
##
## Several indices can name the same table entry, so the scatter adds:
##
## - Metal: one thread per cotangent element, `atomic_float` fetch-add
## - CPU: one thread per table element, accumulating over the index entries
##   (atomic-free: each thread owns one output element)
##
## - The cotangent `g` may be an arbitrary view: element i resolves through the padded 4-axis divmod chain on the cotangent shape.
## - `idx` is a contiguous int32 view. The output is freshly allocated contiguous.
##
## GatherBwdScalars matches the MSL `GatherBwdParams` struct field for field, so one setBytes binds the whole scalar image.

const kGatherBwdMsl* = """
struct GatherBwdParams {
  int32_t gOff;
  int32_t gShape[4];
  int32_t gStrides[4];
  int32_t idxOff;
  int32_t wOff;
  int32_t e;
  int32_t n;
};

kernel void k_gather_bwd(device atomic_float* dw [[buffer(0)]],
                         device const float* g [[buffer(1)]],
                         device const int* idx [[buffer(2)]],
                         constant GatherBwdParams& p [[buffer(3)]],
                         uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    int c3 = i % p.gShape[3];
    int t2 = i / p.gShape[3];
    int c2 = t2 % p.gShape[2];
    int t1 = t2 / p.gShape[2];
    int c1 = t1 % p.gShape[1];
    int c0 = t1 / p.gShape[1];
    int gi = p.gOff + (c0 * p.gStrides[0] + c1 * p.gStrides[1]) +
             c2 * p.gStrides[2] + c3 * p.gStrides[3];
    int entry = i / p.e;
    int col = i - entry * p.e;
    int wrow = idx[p.idxOff + entry];
    atomic_fetch_add_explicit(&dw[p.wOff + wrow * p.e + col], g[gi],
                              memory_order_relaxed);
  }
}
"""

type GatherBwdScalars* = object
  ## Scalar image of one gather backward op:
  ##
  ## - gOff: the cotangent offset in elements
  ## - gShape, gStrides: the padded cotangent shape and strides
  ## - idxOff, wOff: the index and table gradient offsets in elements
  ## - e: the embedding dim, n: the flat element count of the cotangent
  gOff*: int32
  gShape*, gStrides*: array[4, int32]
  idxOff*, wOff*: int32
  e*, n*: int32
  wNumel*: int32
    ## Flat element count of the table gradient and the CPU iteration count

type GatherBwdArgs* = object
  ## Enqueue-time args object of one gather backward op: three 8-byte address
  ## slots (table gradient, cotangent, index) and the scalar image.
  dwAddr*, gAddr*, idxAddr*: uint64
  scalars*: GatherBwdScalars

const
  gatherBwdBufferSlots* = 3
  gatherBwdScalarsSlot* = gatherBwdBufferSlots * 8

proc packGatherBwdArgs*(args: GatherBwdArgs): seq[byte] =
  ## Flat byte image of one recorded gather backward op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `gatherBwdBufferSlots` 8-byte address slots
  ## - the GatherBwdScalars image at `gatherBwdScalarsSlot`
  result = newSeq[byte](gatherBwdScalarsSlot + sizeof(GatherBwdScalars))
  copyMem(result[0].addr, args.dwAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.idxAddr.unsafeAddr, 8)
  copyMem(result[gatherBwdScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(GatherBwdScalars))

proc decodeGatherBwdArgs*(blob: openArray[byte]): GatherBwdArgs =
  ## Inverse of packGatherBwdArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(GatherBwdArgs))

# CPU kernel body
# ------------------------------------------------------------

type GatherBwdEnv*[T] = object
  ## Per-iteration env of the CPU gather backward kernel, one table element per iteration.
  dw*, g*: ptr UncheckedArray[T]
  idx*: ptr UncheckedArray[int32]
  scalars*: GatherBwdScalars

proc gatherBwdBody*[T](t: int32, env: ptr GatherBwdEnv[T]) {.cdecl.} =
  ## CPU port of the k_gather_bwd math, one table element per iteration:
  ## the sum of the cotangent elements whose index names this table entry.
  let sc = env.scalars
  let wrow = t div sc.e
  let col = t - wrow * sc.e
  var acc = T(0)
  let entries = sc.n div sc.e
  for r in 0 ..< entries:
    if env.idx[][sc.idxOff + r] == wrow:
      var c: array[4, int32]
      let i = r * sc.e + col
      var k = i
      for d in countdown(3, 0):
        c[d] = k mod sc.gShape[d]
        k = k div sc.gShape[d]
      var gi = sc.gOff
      for d in 0 .. 3:
        gi += c[d] * sc.gStrides[d]
      acc += env.g[][gi]
  env.dw[][t] = acc
