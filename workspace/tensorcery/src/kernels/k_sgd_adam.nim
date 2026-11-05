# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## In-place optimizer update kernels: sgd and adam, one thread per element.
##
## - SGD, per element:
##   param[i] -= lr * grad[i]
##
## - The caller folds the bias corrections
##   bc1 = 1/(1 - beta1^t) and bc2 = 1/(1 - beta2^t) into the pack at enqueue.
## - Adam, per element:
##   m[i]       = beta1 * m[i] + (1 - beta1) * grad[i]
##   v[i]       = beta2 * v[i] + (1 - beta2) * grad[i]^2
##   param[i]  -= lr * bc1 * m[i] / (sqrt(bc2 * v[i]) + eps)
##
## - Each element is read and written by one thread only, so the in-place update
##   has no cross-thread hazard.
## - The m/v moment tensors persist across iterations. The gradients are fresh per iteration.
## - Operands may be views read through their offsets.
## - SgdScalars / AdamScalars map the MSL `SgdParams` / `AdamParams` structs
##   field for field, so one setBytes binds the whole scalar image.

import std/math

const kSgdMsl* = """
struct SgdParams {
  float lr;
  int32_t pOff;
  int32_t gOff;
  int32_t n;
};

kernel void k_sgd(device float* param [[buffer(0)]],
                  device const float* grad [[buffer(1)]],
                  constant SgdParams& p [[buffer(2)]],
                  uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    param[p.pOff + i] -= p.lr * grad[p.gOff + i];
  }
}
"""

const kAdamMsl* = """
struct AdamParams {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float bc1;
  float bc2;
  int32_t pOff;
  int32_t gOff;
  int32_t mOff;
  int32_t vOff;
  int32_t n;
};

kernel void k_adam(device float* param [[buffer(0)]],
                   device const float* grad [[buffer(1)]],
                   device float* m [[buffer(2)]],
                   device float* v [[buffer(3)]],
                   constant AdamParams& p [[buffer(4)]],
                   uint tid [[thread_position_in_grid]])
{
  int i = int(tid);
  if (i < p.n) {
    float g = grad[p.gOff + i];
    float mi = p.beta1 * m[p.mOff + i] + (1.0 - p.beta1) * g;
    float vi = p.beta2 * v[p.vOff + i] + (1.0 - p.beta2) * g * g;
    m[p.mOff + i] = mi;
    v[p.vOff + i] = vi;
    param[p.pOff + i] -= p.lr * p.bc1 * mi / (sqrt(p.bc2 * vi) + p.eps);
  }
}
"""

type SgdScalars* = object
  ## Scalar image of one sgd op: the learning rate, parameter and gradient offsets in elements, n the flat element count.
  lr*: float32
  pOff*, gOff*, n*: int32

type SgdArgs* = object
  ## Enqueue-time args object of one sgd op: the parameter and gradient addresses
  ## as 8-byte slots, then the scalar image.
  pAddr*, gAddr*: uint64
  scalars*: SgdScalars

type AdamScalars* = object
  ## Scalar image of one adam op:
  ## - the learning rate, decay rates, epsilon and both bias corrections
  ## - the offsets of the parameter, gradient and moment views in elements
  ## - n the flat element count
  lr*, beta1*, beta2*, eps*, bc1*, bc2*: float32
  pOff*, gOff*, mOff*, vOff*, n*: int32

type AdamArgs* = object
  ## Enqueue-time args object of one adam op: the parameter, gradient and moment
  ## addresses as 8-byte slots, then the scalar image.
  pAddr*, gAddr*, mAddr*, vAddr*: uint64
  scalars*: AdamScalars

const
  sgdBufferSlots* = 2
  sgdScalarsSlot* = sgdBufferSlots * 8
  adamBufferSlots* = 4
  adamScalarsSlot* = adamBufferSlots * 8

proc packSgdArgs*(args: SgdArgs): seq[byte] =
  ## Flat byte image of one recorded sgd op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `sgdBufferSlots` 8-byte address slots
  ## - the SgdScalars image at `sgdScalarsSlot`
  result = newSeq[byte](sgdScalarsSlot + sizeof(SgdScalars))
  copyMem(result[0].addr, args.pAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[sgdScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(SgdScalars))

proc decodeSgdArgs*(blob: openArray[byte]): SgdArgs =
  ## Inverse of packSgdArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(SgdArgs))

proc packAdamArgs*(args: AdamArgs): seq[byte] =
  ## Flat byte image of one recorded adam op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `adamBufferSlots` 8-byte address slots
  ## - the AdamScalars image at `adamScalarsSlot`
  result = newSeq[byte](adamScalarsSlot + sizeof(AdamScalars))
  copyMem(result[0].addr, args.pAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.mAddr.unsafeAddr, 8)
  copyMem(result[24].addr, args.vAddr.unsafeAddr, 8)
  copyMem(result[adamScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(AdamScalars))

proc decodeAdamArgs*(blob: openArray[byte]): AdamArgs =
  ## Inverse of packAdamArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(AdamArgs))

# CPU kernel bodies
# ------------------------------------------------------------

type SgdEnv*[T] = object
  ## Per-iteration env of the CPU sgd kernel.
  p*, g*: ptr UncheckedArray[T]
  scalars*: SgdScalars

proc sgdBody*[T](idx: int32, env: ptr SgdEnv[T]) {.cdecl.} =
  ## CPU port of the k_sgd math.
  let sc = env.scalars
  env.p[][sc.pOff + idx] -= T(sc.lr) * env.g[][sc.gOff + idx]

type AdamEnv*[T] = object
  ## Per-iteration env of the CPU adam kernel.
  p*, g*, m*, v*: ptr UncheckedArray[T]
  scalars*: AdamScalars

proc adamBody*[T: SomeFloat](idx: int32, env: ptr AdamEnv[T]) {.cdecl.} =
  ## CPU port of the k_adam math.
  let sc = env.scalars
  let g = env.g[][sc.gOff + idx]
  let mi = sc.beta1 * env.m[][sc.mOff + idx] + (1.0 - sc.beta1) * g
  let vi = sc.beta2 * env.v[][sc.vOff + idx] + (1.0 - sc.beta2) * g * g
  env.m[][sc.mOff + idx] = mi
  env.v[][sc.vOff + idx] = vi
  env.p[][sc.pOff + idx] -= T(sc.lr * sc.bc1) * mi / (sqrt(sc.bc2 * vi) + sc.eps)
