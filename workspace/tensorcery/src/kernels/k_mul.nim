# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal scalar-multiply kernel: output[i] = a[aOff + strided(i)] * scalar.
##
## The kernel reads runtime shapes and strides over a 4-axis divmod chain on the padded shape.
## The operand may be an arbitrary view. The output is freshly allocated contiguous.
##
## MulScalars matches the MSL `MulParams` struct field for field, so one setBytes binds the whole scalar image.

const kMulMsl* = """
struct MulParams {
  int32_t aOff;
  int32_t shape[4];
  int32_t sa[4];
  float scalar;
  int32_t n;
};

kernel void k_mul(device float* output [[buffer(0)]],
                  device const float* a [[buffer(1)]],
                  constant MulParams& p [[buffer(2)]],
                  uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    int c3 = i % p.shape[3];
    int t2 = i / p.shape[3];
    int c2 = t2 % p.shape[2];
    int t1 = t2 / p.shape[2];
    int c1 = t1 % p.shape[1];
    int c0 = t1 / p.shape[1];
    int ia = p.aOff + (c0 * p.sa[0] + c1 * p.sa[1]) + c2 * p.sa[2] + c3 * p.sa[3];
    output[i] = a[ia] * p.scalar;
  }
}
"""

type MulScalars* = object
  ## Scalar image of one scalar-multiply op: operand offset, padded shape and strides, multiplier, n the flat element count.
  aOff*: int32
  shape*: array[4, int32]
  sa*: array[4, int32]
  scalar*: float32
  n*: int32

type MulArgs* = object
  ## Enqueue-time args object of one scalar-multiply op: output and a device addresses as 8-byte slots, then the scalar image.
  outAddr*, aAddr*: uint64
  scalars*: MulScalars

const
  mulBufferSlots* = 2
  mulScalarsSlot* = mulBufferSlots * 8

proc packMulArgs*(args: MulArgs): seq[byte] =
  ## Flat byte image of one recorded scalar-multiply op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `mulBufferSlots` 8-byte device address slots
  ## - the MulScalars image at `mulScalarsSlot`
  result = newSeq[byte](mulScalarsSlot + sizeof(MulScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.aAddr.unsafeAddr, 8)
  copyMem(result[mulScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(MulScalars))

proc decodeMulArgs*(blob: openArray[byte]): MulArgs =
  ## Inverse of packMulArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(MulArgs))
