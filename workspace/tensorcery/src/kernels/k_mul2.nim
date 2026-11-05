# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal elementwise tensor-product kernel: output[i] = a[aOff + strided(i)] * b[bOff + strided(i)].
##
## The kernel reads runtime shapes and strides over a 4-axis divmod chain on the padded shape.
## Both operands may be arbitrary views. The output is freshly allocated contiguous.
##
## Mul2Scalars matches the MSL `Mul2Params` struct field for field, so one setBytes binds the whole scalar image.

const kMul2Msl* = """
struct Mul2Params {
  int32_t aOff;
  int32_t bOff;
  int32_t shape[4];
  int32_t sa[4];
  int32_t sb[4];
  int32_t n;
};

kernel void k_mul2(device float* output [[buffer(0)]],
                   device const float* a [[buffer(1)]],
                   device const float* b [[buffer(2)]],
                   constant Mul2Params& p [[buffer(3)]],
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
    int ib = p.bOff + (c0 * p.sb[0] + c1 * p.sb[1]) + c2 * p.sb[2] + c3 * p.sb[3];
    output[i] = a[ia] * b[ib];
  }
}
"""

type Mul2Scalars* = object
  ## Scalar image of one tensor-product op: operand offsets in elements, shape and strides
  ## padded to 4 axes, n the flat element count.
  aOff*, bOff*: int32
  shape*: array[4, int32]
  sa*, sb*: array[4, int32]
  n*: int32

type Mul2Args* = object
  ## Enqueue-time args object of one tensor-product op: the output, a and b device
  ## addresses as 8-byte slots, then the scalar image.
  outAddr*, aAddr*, bAddr*: uint64
  scalars*: Mul2Scalars

const
  mul2BufferSlots* = 3
  mul2ScalarsSlot* = mul2BufferSlots * 8

proc packMul2Args*(args: Mul2Args): seq[byte] =
  ## Flat byte image of one recorded tensor-product op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `mul2BufferSlots` 8-byte device address slots
  ## - the Mul2Scalars image at `mul2ScalarsSlot`
  result = newSeq[byte](mul2ScalarsSlot + sizeof(Mul2Scalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.aAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.bAddr.unsafeAddr, 8)
  copyMem(result[mul2ScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(Mul2Scalars))

proc decodeMul2Args*(blob: openArray[byte]): Mul2Args =
  ## Inverse of packMul2Args: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(Mul2Args))
