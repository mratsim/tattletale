# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tanh-approximate gelu kernel (the GPT-2 gelu):
##   output[i] = 0.5 * x * (1 + tanh(u))
##   with u = sqrt(2/pi) * (x + 0.044715 * x^3)
##
## The input may be an arbitrary view. The output is freshly allocated contiguous.
##
## GeluScalars matches the MSL `GeluParams` struct field for field, so one setBytes
## binds the whole scalar image.

import std/math

const kGeluMsl* = """
struct GeluParams {
  int32_t aOff;
  int32_t n;
};

kernel void k_gelu(device float* output [[buffer(0)]],
                   device const float* a [[buffer(1)]],
                   constant GeluParams& p [[buffer(2)]],
                   uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    float x = a[p.aOff + i];
    float u = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    output[i] = 0.5 * x * (1.0 + tanh(u));
  }
}
"""

type GeluScalars* = object
  ## Scalar image of one gelu op: input offset in elements, n the flat element count.
  aOff*, n*: int32

type GeluArgs* = object
  ## Enqueue-time args object of one gelu op: the output and input addresses as 8-byte slots, then the scalar image.
  outAddr*, aAddr*: uint64
  scalars*: GeluScalars

const
  geluBufferSlots* = 2
  geluScalarsSlot* = geluBufferSlots * 8

proc packGeluArgs*(args: GeluArgs): seq[byte] =
  ## Flat byte image of one recorded gelu op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `geluBufferSlots` 8-byte address slots
  ## - the GeluScalars image at `geluScalarsSlot`
  result = newSeq[byte](geluScalarsSlot + sizeof(GeluScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.aAddr.unsafeAddr, 8)
  copyMem(result[geluScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(GeluScalars))

proc decodeGeluArgs*(blob: openArray[byte]): GeluArgs =
  ## Inverse of packGeluArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(GeluArgs))

# CPU kernel body
# ------------------------------------------------------------

type GeluEnv*[T] = object
  ## Per-iteration env of the CPU gelu kernel.
  outp*, a*: ptr UncheckedArray[T]
  scalars*: GeluScalars

proc geluBody*[T: SomeFloat](idx: int32, env: ptr GeluEnv[T]) {.cdecl.} =
  ## CPU port of the k_gelu math.
  let sc = env.scalars
  let x = env.a[][sc.aOff + idx]
  let u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
  env.outp[][idx] = 0.5 * x * (1.0 + tanh(u))
