# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Last-axis layernorm kernel: one thread per set of d elements.
##
## For element i, r = i div d and c = i mod d:
##   mean   = sum_c x[r, c] / d
##   var    = sum_c (x[r, c] - mean)^2 / d
##   out[i] = (x[r, c] - mean) / sqrt(var + eps) * w[c] + b[c]
##
## - The input may be an arbitrary view: the base address of each d-element set
##   resolves through the padded 4-axis divmod chain on the input shape.
## - The inner loop steps by the last-axis stride.
## - `w` and `b` are contiguous d-element parameter views. The output is freshly
##   allocated contiguous.
## - LayerNormScalars matches the MSL `LayerNormParams` struct field for field, so one setBytes binds the whole scalar image.

import std/math

const kLayerNormMsl* = """
struct LayerNormParams {
  int32_t xOff;
  int32_t wOff;
  int32_t bOff;
  int32_t xShape[4];
  int32_t xs[4];
  int32_t d;
  int32_t rows;
  int32_t xStep;
  float eps;
};

kernel void k_layernorm(device float* output [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device const float* w [[buffer(2)]],
                        device const float* b [[buffer(3)]],
                        constant LayerNormParams& p [[buffer(4)]],
                        uint tid [[thread_position_in_grid]])
{
  int r = int(tid);
  if (r < p.rows) {
    int i0 = r * p.d;
    int c3 = i0 % p.xShape[3];
    int t2 = i0 / p.xShape[3];
    int c2 = t2 % p.xShape[2];
    int t1 = t2 / p.xShape[2];
    int c1 = t1 % p.xShape[1];
    int c0 = t1 / p.xShape[1];
    int base = p.xOff + (c0 * p.xs[0] + c1 * p.xs[1]) + c2 * p.xs[2] + c3 * p.xs[3];
    float mean = 0.0;
    for (int c = 0; c < p.d; c++) {
      mean += x[base + c * p.xStep];
    }
    mean /= float(p.d);
    float var = 0.0;
    for (int c = 0; c < p.d; c++) {
      float dv = x[base + c * p.xStep] - mean;
      var += dv * dv;
    }
    var /= float(p.d);
    float scale = 1.0 / sqrt(var + p.eps);
    for (int c = 0; c < p.d; c++) {
      output[i0 + c] = (x[base + c * p.xStep] - mean) * scale * w[p.wOff + c] + b[p.bOff + c];
    }
  }
}
"""

type LayerNormScalars* = object
  ## Scalar image of one layernorm op:
  ## - the input, scale and shift offsets in elements
  ## - the padded input shape and strides
  ## - d the normalized axis size and the d-element set count
  ## - xStep the normalized-axis stride of the input view and eps the variance epsilon
  xOff*, wOff*, bOff*: int32
  xShape*, xs*: array[4, int32]
  d*, rows*: int32
  xStep*: int32
  eps*: float32

type LayerNormArgs* = object
  ## Enqueue-time args object of one layernorm op: the output, input, scale and shift
  ## addresses as 8-byte slots, then the scalar image.
  outAddr*, xAddr*, wAddr*, bAddr*: uint64
  scalars*: LayerNormScalars

const
  layerNormBufferSlots* = 4
  layerNormScalarsSlot* = layerNormBufferSlots * 8

proc packLayerNormArgs*(args: LayerNormArgs): seq[byte] =
  ## Flat byte image of one recorded layernorm op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `layerNormBufferSlots` 8-byte address slots
  ## - the LayerNormScalars image at `layerNormScalarsSlot`
  result = newSeq[byte](layerNormScalarsSlot + sizeof(LayerNormScalars))
  copyMem(result[0].addr, args.outAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.xAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.wAddr.unsafeAddr, 8)
  copyMem(result[24].addr, args.bAddr.unsafeAddr, 8)
  copyMem(result[layerNormScalarsSlot].addr, args.scalars.unsafeAddr, sizeof(LayerNormScalars))

proc decodeLayerNormArgs*(blob: openArray[byte]): LayerNormArgs =
  ## Inverse of packLayerNormArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(LayerNormArgs))

# CPU kernel body
# ------------------------------------------------------------

type LayerNormEnv*[T] = object
  ## Per-iteration env of the CPU layernorm kernel, one d-element set per iteration.
  outp*, x*, w*, b*: ptr UncheckedArray[T]
  scalars*: LayerNormScalars

proc layernormBody*[T: SomeFloat](r: int32, env: ptr LayerNormEnv[T]) {.cdecl.} =
  ## CPU port of the k_layernorm math, one set of d elements per iteration.
  let sc = env.scalars
  let i0 = r * sc.d
  var c: array[4, int32]
  var t = i0
  for k in countdown(3, 0):
    c[k] = t mod sc.xShape[k]
    t = t div sc.xShape[k]
  var base = sc.xOff
  for k in 0 .. 3:
    base += c[k] * sc.xs[k]
  var mean = T(0)
  for j in 0 ..< sc.d:
    mean += env.x[][base + j * sc.xStep]
  mean /= T(sc.d)
  var vr = T(0)
  for j in 0 ..< sc.d:
    let dv = env.x[][base + j * sc.xStep] - mean
    vr += dv * dv
  vr /= T(sc.d)
  let scale = T(1) / sqrt(vr + T(sc.eps))
  for j in 0 ..< sc.d:
    env.outp[][i0 + j] = (env.x[][base + j * sc.xStep] - mean) * scale *
        env.w[][sc.wOff + j] + env.b[][sc.bOff + j]
