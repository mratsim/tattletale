# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Elementwise select kernel: output[i] = cond[i] != 0 ? a[i] : b[i].
##
## All operands are same-shape contiguous views read through their offsets.
## The output is freshly allocated contiguous. Causal masking builds on this op:
## the mask is 1 on kept elements and 0 on masked ones. The masked branch
## supplies the sentinel value.
##
## WhereScalars matches the MSL `WhereParams` struct field for field, so one setBytes
## binds the whole scalar image.

const kWhereMsl* = """
struct WhereParams {
  int32_t condOff;
  int32_t aOff;
  int32_t bOff;
  int32_t n;
};

kernel void k_where(device float* output [[buffer(0)]],
                    device const float* cond [[buffer(1)]],
                    device const float* a [[buffer(2)]],
                    device const float* b [[buffer(3)]],
                    constant WhereParams& p [[buffer(4)]],
                    uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    if (cond[p.condOff + i] != 0.0) {
      output[i] = a[p.aOff + i];
    } else {
      output[i] = b[p.bOff + i];
    }
  }
}
"""

type WhereScalars* = object
  ## Scalar image of one where op: condition and operand offsets in elements,
  ## n the flat element count.
  condOff*, aOff*, bOff*, n*: int32

type WhereArgs* = object
  ## Enqueue-time args object of one where op: the output, condition and operand
  ## addresses as 8-byte slots, then the scalar image.
  outAddr*, condAddr*, aAddr*, bAddr*: uint64
  scalars*: WhereScalars

const
  whereBufferSlots* = 4
  whereScalarsSlot* = whereBufferSlots * 8

proc packWhereArgs*(args: WhereArgs): seq[byte] =
  ## Flat byte image of one recorded where op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `whereBufferSlots` 8-byte address slots
  ## - the WhereScalars image at `whereScalarsSlot`
  result = newSeq[byte](whereScalarsSlot + sizeof(WhereScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.condAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.aAddr.unsafeAddr, 8)
  copyMem(result[24].addr, args.bAddr.unsafeAddr, 8)
  copyMem(result[whereScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(WhereScalars))

proc decodeWhereArgs*(blob: openArray[byte]): WhereArgs =
  ## Inverse of packWhereArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(WhereArgs))

# CPU kernel body
# ------------------------------------------------------------

type WhereEnv*[T] = object
  ## Per-iteration env of the CPU where kernel.
  outp*, cond*, a*, b*: ptr UncheckedArray[T]
  scalars*: WhereScalars

proc whereBody*[T](idx: int32, env: ptr WhereEnv[T]) {.cdecl.} =
  ## CPU port of the k_where select.
  let sc = env.scalars
  if env.cond[][sc.condOff + idx] != T(0):
    env.outp[][idx] = env.a[][sc.aOff + idx]
  else:
    env.outp[][idx] = env.b[][sc.bOff + idx]
