# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal elementwise add kernel: output[i] = a[aOff + strided(i)] + b[bOff + strided(i)].
##
## The kernel reads runtime shapes and strides over a 4-axis divmod chain on the padded shape.
## Both operands may be arbitrary views. The output is freshly allocated contiguous.
##
## AddScalars matches the MSL `AddParams` struct field for field, so one setBytes binds the whole scalar image.

const kAddMsl* = """
struct AddParams {
  int32_t aOff;
  int32_t bOff;
  int32_t shape[4];
  int32_t sa[4];
  int32_t sb[4];
  int32_t n;
};

kernel void k_add(device float* output [[buffer(0)]],
                  device const float* a [[buffer(1)]],
                  device const float* b [[buffer(2)]],
                  constant AddParams& p [[buffer(3)]],
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
    output[i] = a[ia] + b[ib];
  }
}
"""

type AddScalars* = object
  ## Scalar image of one add op: operand offsets in elements, shape and strides
  ## padded to 4 axes, n the flat element count.
  aOff*, bOff*: int32
  shape*: array[4, int32]
  sa*, sb*: array[4, int32]
  n*: int32

type AddArgs* = object
  ## Enqueue-time args object of one add op: the output, a and b device
  ## addresses as 8-byte slots, then the scalar image.
  outAddr*, aAddr*, bAddr*: uint64
  scalars*: AddScalars

const
  addBufferSlots* = 3
  addScalarsSlot* = addBufferSlots * 8

proc packAddArgs*(args: AddArgs): seq[byte] =
  ## Flat byte image of one recorded add op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `addBufferSlots` 8-byte device address slots
  ## - the AddScalars image at `addScalarsSlot`
  result = newSeq[byte](addScalarsSlot + sizeof(AddScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.aAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.bAddr.unsafeAddr, 8)
  copyMem(result[addScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(AddScalars))
