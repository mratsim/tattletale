# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tanh-approximate gelu backward kernel, elementwise:
##   u = k * (x + 0.044715 * x^3), k = sqrt(2/pi)
##   dx[i] = g[i] * (0.5 * (1 + tanh u) + 0.5 * x * (1 - tanh^2 u) * k * (1 + 3 * 0.044715 * x^2))
##
## - The cotangent `g` may be an arbitrary view: element i resolves through the padded 4-axis divmod chain on the cotangent shape.
## - The input `x` reads linearly from its offset, matching the k_gelu forward.
## - The output is freshly allocated contiguous.
##
## GeluBwdScalars matches the MSL `GeluBwdParams` struct field for field, so one setBytes binds the whole scalar image.

import std/math

const kGeluBwdMsl* = """
struct GeluBwdParams {
  int32_t gOff;
  int32_t gShape[4];
  int32_t gStrides[4];
  int32_t aOff;
  int32_t n;
};

kernel void k_gelu_bwd(device float* dx [[buffer(0)]],
                       device const float* g [[buffer(1)]],
                       device const float* x [[buffer(2)]],
                       constant GeluBwdParams& p [[buffer(3)]],
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
    int gi = p.gOff + (c0 * p.gStrides[0] + c1 * p.gStrides[1]) + c2 * p.gStrides[2] + c3 * p.gStrides[3];
    float xv = x[p.aOff + i];
    float u = 0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
    float th = tanh(u);
    float dv = 0.5 * (1.0 + th) + 0.5 * xv * (1.0 - th * th) * 0.7978845608028654 *
               (1.0 + 3.0 * 0.044715 * xv * xv);
    dx[i] = g[gi] * dv;
  }
}
"""

type GeluBwdScalars* = object
  ## Scalar image of one gelu backward op:
  ##
  ## - gOff: the cotangent offset in elements
  ## - gShape, gStrides: the padded cotangent shape and strides
  ## - aOff: the input offset in elements
  ## - n: the flat element count
  gOff*: int32
  gShape*, gStrides*: array[4, int32]
  aOff*: int32
  n*: int32

type GeluBwdArgs* = object
  ## Enqueue-time args object of one gelu backward op: three 8-byte address
  ## slots (output, cotangent, input) and the scalar image.
  dxAddr*, gAddr*, xAddr*: uint64
  scalars*: GeluBwdScalars

const
  geluBwdBufferSlots* = 3
  geluBwdScalarsSlot* = geluBwdBufferSlots * 8

proc packGeluBwdArgs*(args: GeluBwdArgs): seq[byte] =
  ## Flat byte image of one recorded gelu backward op, marshaled once at enqueue
  ## and decoded at eval.
  ##
  ## Returns:
  ## - `geluBwdBufferSlots` 8-byte address slots
  ## - the GeluBwdScalars image at `geluBwdScalarsSlot`
  result = newSeq[byte](geluBwdScalarsSlot + sizeof(GeluBwdScalars))
  copyMem(result[0].addr, args.dxAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.xAddr.unsafeAddr, 8)
  copyMem(result[geluBwdScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(GeluBwdScalars))

proc decodeGeluBwdArgs*(blob: openArray[byte]): GeluBwdArgs =
  ## Inverse of packGeluBwdArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(GeluBwdArgs))

# CPU kernel body
# ------------------------------------------------------------

type GeluBwdEnv*[T] = object
  ## Per-iteration env of the CPU gelu backward kernel.
  dx*, g*, x*: ptr UncheckedArray[T]
  scalars*: GeluBwdScalars

proc geluBwdBody*[T: SomeFloat](idx: int32, env: ptr GeluBwdEnv[T]) {.cdecl.} =
  ## CPU port of the k_gelu_bwd math.
  let sc = env.scalars
  var c: array[4, int32]
  var t = idx
  for k in countdown(3, 0):
    c[k] = t mod sc.gShape[k]
    t = t div sc.gShape[k]
  var gi = sc.gOff
  for k in 0 .. 3:
    gi += c[k] * sc.gStrides[k]
  let xv = env.x[][sc.aOff + idx]
  let u = 0.7978845608028654 * (xv + 0.044715 * xv * xv * xv)
  let th = tanh(u)
  let dv = 0.5 * (1.0 + th) + 0.5 * xv * (1.0 - th * th) * 0.7978845608028654 *
      (1.0 + 3.0 * 0.044715 * xv * xv)
  env.dx[][idx] = env.g[][gi] * T(dv)
