# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedding gather kernel: output[i] = weight[idx[i div e] * e + i mod e].
##
## - `weight`: the rank-2 embedding table (vocabulary, embedding dim), contiguous
## - `idx`: flat table indices, one int32 per output entry, contiguous
## - output: freshly allocated contiguous, shape idx.shape + [e]
##
## GatherScalars matches the MSL `GatherParams` struct field for field, so one setBytes
## binds the whole scalar image.

const kGatherMsl* = """
struct GatherParams {
  int32_t wOff;
  int32_t idxOff;
  int32_t e;
  int32_t n;
};

kernel void k_gather(device float* output [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device const int* idx [[buffer(2)]],
                     constant GatherParams& p [[buffer(3)]],
                     uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    int row = i / p.e;
    int col = i - row * p.e;
    int wrow = idx[p.idxOff + row];
    output[i] = w[p.wOff + wrow * p.e + col];
  }
}
"""

type GatherScalars* = object
  ## Scalar image of one gather op: table and index offsets in elements,
  ## e the embedding dim, n the flat element count of the output.
  wOff*, idxOff*, e*, n*: int32

type GatherArgs* = object
  ## Enqueue-time args object of one gather op: the output, weight and index
  ## addresses as 8-byte slots, then the scalar image.
  outAddr*, wAddr*, idxAddr*: uint64
  scalars*: GatherScalars

const
  gatherBufferSlots* = 3
  gatherScalarsSlot* = gatherBufferSlots * 8

proc packGatherArgs*(args: GatherArgs): seq[byte] =
  ## Flat byte image of one recorded gather op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `gatherBufferSlots` 8-byte address slots
  ## - the GatherScalars image at `gatherScalarsSlot`
  result = newSeq[byte](gatherScalarsSlot + sizeof(GatherScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.wAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.idxAddr.unsafeAddr, 8)
  copyMem(result[gatherScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(GatherScalars))

proc decodeGatherArgs*(blob: openArray[byte]): GatherArgs =
  ## Inverse of packGatherArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(GatherArgs))

# CPU kernel body
# ------------------------------------------------------------

type GatherEnv*[T] = object
  ## Per-iteration env of the CPU gather kernel.
  outp*, w*: ptr UncheckedArray[T]
  idx*: ptr UncheckedArray[int32]
  scalars*: GatherScalars

proc gatherBody*[T](idx: int32, env: ptr GatherEnv[T]) {.cdecl.} =
  ## CPU port of the k_gather math: output[i] = weight[idx[i div e] * e + i mod e]
  let sc = env.scalars
  let entry = idx div sc.e
  let col = idx - entry * sc.e
  let srcRow = env.idx[][sc.idxOff + entry]
  env.outp[][idx] = env.w[][sc.wOff + srcRow * sc.e + col]
