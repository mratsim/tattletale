# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal ones-fill kernel: output[i] = 1.
##
## Grad-chain seed primitive: backward() seeds the cotangent map with ones over the loss shape,
## so differentiating a tensor means differentiating the sum of its elements.
## The output is freshly allocated contiguous.
##
## FillScalars matches the MSL `FillParams` struct field for field, so one setBytes binds the whole scalar image.

const kFillMsl* = """
struct FillParams {
  int32_t shape[4];
  int32_t n;
};

kernel void k_fill(device float* output [[buffer(0)]],
                   constant FillParams& p [[buffer(1)]],
                   uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    output[i] = 1.0;
  }
}
"""

type FillScalars* = object
  ## Scalar image of one fill op: shape padded to 4 axes, n the flat element count.
  shape*: array[4, int32]
  n*: int32

type FillArgs* = object
  ## Enqueue-time args object of one fill op: the output device address as one
  ## 8-byte slot, then the scalar image.
  outAddr*: uint64
  scalars*: FillScalars

const
  fillBufferSlots* = 1
  fillScalarsSlot* = fillBufferSlots * 8

proc packFillArgs*(args: FillArgs): seq[byte] =
  ## Flat byte image of one recorded fill op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `fillBufferSlots` 8-byte device address slots
  ## - the FillScalars image at `fillScalarsSlot`
  result = newSeq[byte](fillScalarsSlot + sizeof(FillScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[fillScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(FillScalars))

proc decodeFillArgs*(blob: openArray[byte]): FillArgs =
  ## Inverse of packFillArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(FillArgs))
