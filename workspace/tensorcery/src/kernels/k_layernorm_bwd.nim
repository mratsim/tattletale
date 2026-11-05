# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Last-axis layernorm backward kernels: dX per set, dW and dB per parameter element.
##
## With xhat = (x - mean) / sqrt(var + eps) per set, g the cotangent of the output
## and dxhat[c] = g[c] * w[c], mean and var over the normalized axis as k_layernorm computes them:
##   dx[r*d + c] = invstd_r * (dxhat[r, c] - sum_c' dxhat[r, c'] / d - xhat[r, c] * sum_c' dxhat[r, c'] * xhat[r, c'] / d)
##
##   dw[c] = sum_r g[r, c] * xhat[r, c]
##
##   db[c] = sum_r g[r, c]
##
## | kernel               | thread budget                                                               | writes             |
## | -------------------- | --------------------------------------------------------------------------- | ------------------ |
## | `k_layernorm_bwd_dx` | one thread per d-element set, recomputes the set mean and var               | the dX element set |
## | `k_layernorm_bwd_w`  | one thread per parameter element, recomputes each set mean and var per scan | one dW element     |
## | `k_layernorm_bwd_b`  | one thread per parameter element, scans the cotangent only                  | one dB element     |
##
## - The cotangent `g` and the input `x` may be arbitrary views: the base of each d-element set
##   resolves through the padded 4-axis divmod chains on the two shapes, and the inner loops advance by `gStep` / `xStep`.
## - `w` is a contiguous d-element parameter view and the outputs are freshly allocated contiguous.
## - LayerNormBwdScalars matches the MSL `LayerNormBwdParams` struct field for field, so one setBytes binds the whole scalar image.

import std/math

const kLayerNormBwdMsl* = """
struct LayerNormBwdParams {
  int32_t gOff;
  int32_t wOff;
  int32_t xOff;
  int32_t gShape[4];
  int32_t gStrides[4];
  int32_t gStep;
  int32_t xShape[4];
  int32_t xs[4];
  int32_t xStep;
  int32_t d;
  int32_t rows;
  float eps;
};

kernel void k_layernorm_bwd_dx(device float* dx [[buffer(0)]],
                               device const float* g [[buffer(1)]],
                               device const float* x [[buffer(2)]],
                               device const float* w [[buffer(3)]],
                               constant LayerNormBwdParams& p [[buffer(4)]],
                               uint tid [[thread_position_in_grid]])
{
  int r = int(tid);
  if (r < p.rows) {
    int i0 = r * p.d;
    int gc3 = i0 % p.gShape[3];
    int gt2 = i0 / p.gShape[3];
    int gc2 = gt2 % p.gShape[2];
    int gt1 = gt2 / p.gShape[2];
    int gc1 = gt1 % p.gShape[1];
    int gc0 = gt1 / p.gShape[1];
    int gbase = p.gOff + (gc0 * p.gStrides[0] + gc1 * p.gStrides[1]) +
                gc2 * p.gStrides[2] + gc3 * p.gStrides[3];
    int xc3 = i0 % p.xShape[3];
    int xt2 = i0 / p.xShape[3];
    int xc2 = xt2 % p.xShape[2];
    int xt1 = xt2 / p.xShape[2];
    int xc1 = xt1 % p.xShape[1];
    int xc0 = xt1 / p.xShape[1];
    int xbase = p.xOff + (xc0 * p.xs[0] + xc1 * p.xs[1]) + xc2 * p.xs[2] + xc3 * p.xs[3];
    float mean = 0.0;
    for (int c = 0; c < p.d; c++) {
      mean += x[xbase + c * p.xStep];
    }
    mean /= float(p.d);
    float var = 0.0;
    for (int c = 0; c < p.d; c++) {
      float dv = x[xbase + c * p.xStep] - mean;
      var += dv * dv;
    }
    var /= float(p.d);
    float invstd = 1.0 / sqrt(var + p.eps);
    float sum1 = 0.0;
    float sum2 = 0.0;
    for (int c = 0; c < p.d; c++) {
      float dxh = g[gbase + c * p.gStep] * w[p.wOff + c];
      float xh = (x[xbase + c * p.xStep] - mean) * invstd;
      sum1 += dxh;
      sum2 += dxh * xh;
    }
    for (int c = 0; c < p.d; c++) {
      float dxh = g[gbase + c * p.gStep] * w[p.wOff + c];
      float xh = (x[xbase + c * p.xStep] - mean) * invstd;
      dx[i0 + c] = invstd * (dxh - sum1 / float(p.d) - xh * sum2 / float(p.d));
    }
  }
}

kernel void k_layernorm_bwd_w(device float* dw [[buffer(0)]],
                              device const float* g [[buffer(1)]],
                              device const float* x [[buffer(2)]],
                              constant LayerNormBwdParams& p [[buffer(3)]],
                              uint tid [[thread_position_in_grid]])
{
  int c = int(tid);
  if (c < p.d) {
    float acc = 0.0;
    for (int r = 0; r < p.rows; r++) {
      int s0 = r * p.d;
      int xc3 = s0 % p.xShape[3];
      int xt2 = s0 / p.xShape[3];
      int xc2 = xt2 % p.xShape[2];
      int xt1 = xt2 / p.xShape[2];
      int xc1 = xt1 % p.xShape[1];
      int xc0 = xt1 / p.xShape[1];
      int xbase = p.xOff + (xc0 * p.xs[0] + xc1 * p.xs[1]) + xc2 * p.xs[2] + xc3 * p.xs[3];
      int i0 = s0 + c;
      int gc3 = i0 % p.gShape[3];
      int gt2 = i0 / p.gShape[3];
      int gc2 = gt2 % p.gShape[2];
      int gt1 = gt2 / p.gShape[2];
      int gc1 = gt1 % p.gShape[1];
      int gc0 = gt1 / p.gShape[1];
      int gi = p.gOff + (gc0 * p.gStrides[0] + gc1 * p.gStrides[1]) +
               gc2 * p.gStrides[2] + gc3 * p.gStrides[3];
      float mean = 0.0;
      for (int j = 0; j < p.d; j++) {
        mean += x[xbase + j * p.xStep];
      }
      mean /= float(p.d);
      float var = 0.0;
      for (int j = 0; j < p.d; j++) {
        float dv = x[xbase + j * p.xStep] - mean;
        var += dv * dv;
      }
      var /= float(p.d);
      float xh = (x[xbase + c * p.xStep] - mean) / sqrt(var + p.eps);
      acc += g[gi] * xh;
    }
    dw[c] = acc;
  }
}

kernel void k_layernorm_bwd_b(device float* db [[buffer(0)]],
                              device const float* g [[buffer(1)]],
                              constant LayerNormBwdParams& p [[buffer(2)]],
                              uint tid [[thread_position_in_grid]])
{
  int c = int(tid);
  if (c < p.d) {
    float acc = 0.0;
    for (int r = 0; r < p.rows; r++) {
      int i0 = r * p.d + c;
      int gc3 = i0 % p.gShape[3];
      int gt2 = i0 / p.gShape[3];
      int gc2 = gt2 % p.gShape[2];
      int gt1 = gt2 / p.gShape[2];
      int gc1 = gt1 % p.gShape[1];
      int gc0 = gt1 / p.gShape[1];
      int gi = p.gOff + (gc0 * p.gStrides[0] + gc1 * p.gStrides[1]) +
               gc2 * p.gStrides[2] + gc3 * p.gStrides[3];
      acc += g[gi];
    }
    db[c] = acc;
  }
}
"""

type LayerNormBwdScalars* = object
  ## Scalar image of one layernorm backward op, shared by the three kernels:
  ##
  ## - gOff, wOff, xOff: the cotangent, scale and input offsets in elements
  ## - gShape, gStrides: the padded cotangent shape and strides
  ## - gStep: the cotangent's normalized-axis stride
  ## - xShape, xs: the padded input shape and strides
  ##
  ## - xStep: the input's normalized-axis stride
  ## - d: the normalized axis size and the d-element set count
  ## - eps: the variance epsilon of the k_layernorm forward
  gOff*, wOff*, xOff*: int32
  gShape*, gStrides*: array[4, int32]
  gStep*: int32
  xShape*, xs*: array[4, int32]
  xStep*, d*, rows*: int32
  eps*: float32

type LayerNormBwdDxArgs* = object
  ## Enqueue-time args object of the dX kernel: four 8-byte address slots
  ## (output, cotangent, input, scale) and the scalar image.
  dxAddr*, gAddr*, xAddr*, wAddr*: uint64
  scalars*: LayerNormBwdScalars

type LayerNormBwdWArgs* = object
  ## Enqueue-time args object of the dW kernel: three 8-byte address slots
  ## (output, cotangent, input) and the scalar image.
  dwAddr*, gAddr*, xAddr*: uint64
  scalars*: LayerNormBwdScalars

type LayerNormBwdBArgs* = object
  ## Enqueue-time args object of the dB kernel: two 8-byte address slots
  ## (output, cotangent) and the scalar image.
  dbAddr*, gAddr*: uint64
  scalars*: LayerNormBwdScalars

const
  layerNormBwdDxBufferSlots* = 4
  layerNormBwdWBufferSlots* = 3
  layerNormBwdBBufferSlots* = 2
  layerNormBwdDxScalarsSlot* = layerNormBwdDxBufferSlots * 8
  layerNormBwdWScalarsSlot* = layerNormBwdWBufferSlots * 8
  layerNormBwdBScalarsSlot* = layerNormBwdBBufferSlots * 8

proc packLayerNormBwdDxArgs*(args: LayerNormBwdDxArgs): seq[byte] =
  ## Flat byte image of one recorded layernorm backward dX op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `layerNormBwdDxBufferSlots` 8-byte address slots
  ## - the LayerNormBwdScalars image at `layerNormBwdDxScalarsSlot`
  result = newSeq[byte](layerNormBwdDxScalarsSlot + sizeof(LayerNormBwdScalars))
  copyMem(result[0].addr, args.dxAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.xAddr.unsafeAddr, 8)
  copyMem(result[24].addr, args.wAddr.unsafeAddr, 8)
  copyMem(result[layerNormBwdDxScalarsSlot].addr, args.scalars.unsafeAddr,
      sizeof(LayerNormBwdScalars))

proc packLayerNormBwdWArgs*(args: LayerNormBwdWArgs): seq[byte] =
  ## Flat byte image of one recorded layernorm backward dW op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `layerNormBwdWBufferSlots` 8-byte address slots
  ## - the LayerNormBwdScalars image at `layerNormBwdWScalarsSlot`
  result = newSeq[byte](layerNormBwdWScalarsSlot + sizeof(LayerNormBwdScalars))
  copyMem(result[0].addr, args.dwAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[16].addr, args.xAddr.unsafeAddr, 8)
  copyMem(result[layerNormBwdWScalarsSlot].addr, args.scalars.unsafeAddr,
      sizeof(LayerNormBwdScalars))

proc packLayerNormBwdBArgs*(args: LayerNormBwdBArgs): seq[byte] =
  ## Flat byte image of one recorded layernorm backward dB op, marshaled once at enqueue and decoded at eval.
  ##
  ## Returns:
  ## - `layerNormBwdBBufferSlots` 8-byte address slots
  ## - the LayerNormBwdScalars image at `layerNormBwdBScalarsSlot`
  result = newSeq[byte](layerNormBwdBScalarsSlot + sizeof(LayerNormBwdScalars))
  copyMem(result[0].addr, args.dbAddr.unsafeAddr, 8)
  copyMem(result[8].addr, args.gAddr.unsafeAddr, 8)
  copyMem(result[layerNormBwdBScalarsSlot].addr, args.scalars.unsafeAddr,
      sizeof(LayerNormBwdScalars))

# CPU kernel bodies
# ------------------------------------------------------------

type LayerNormBwdDxEnv*[T] = object
  ## Per-iteration env of the CPU layernorm backward dX kernel, one set per iteration.
  dx*, g*, x*, w*: ptr UncheckedArray[T]
  scalars*: LayerNormBwdScalars

proc layerNormSetBase(gShape, gStrides: array[4, int32], off: int32, i0: int32): int =
  ## Base offset of the d-element set at linear set index i0 through a padded shape/stride pair.
  var c: array[4, int32]
  var t = i0
  for k in countdown(3, 0):
    c[k] = t mod gShape[k]
    t = t div gShape[k]
  result = off
  for k in 0 .. 3:
    result += c[k] * gStrides[k]

proc layernormBwdDxBody*[T: SomeFloat](r: int32, env: ptr LayerNormBwdDxEnv[T]) {.cdecl.} =
  ## CPU port of the k_layernorm_bwd_dx math, one set of d elements per iteration.
  let sc = env.scalars
  let i0 = r * sc.d
  let gbase = layerNormSetBase(sc.gShape, sc.gStrides, sc.gOff, i0)
  let xbase = layerNormSetBase(sc.xShape, sc.xs, sc.xOff, i0)
  var mean = T(0)
  for j in 0 ..< sc.d:
    mean += env.x[][xbase + j * sc.xStep]
  mean /= T(sc.d)
  var vr = T(0)
  for j in 0 ..< sc.d:
    let dv = env.x[][xbase + j * sc.xStep] - mean
    vr += dv * dv
  vr /= T(sc.d)
  let invstd = T(1) / sqrt(vr + T(sc.eps))
  var sum1 = T(0)
  var sum2 = T(0)
  for j in 0 ..< sc.d:
    let dxh = env.g[][gbase + j * sc.gStep] * env.w[][sc.wOff + j]
    let xh = (env.x[][xbase + j * sc.xStep] - mean) * invstd
    sum1 += dxh
    sum2 += dxh * xh
  for j in 0 ..< sc.d:
    let dxh = env.g[][gbase + j * sc.gStep] * env.w[][sc.wOff + j]
    let xh = (env.x[][xbase + j * sc.xStep] - mean) * invstd
    env.dx[][i0 + j] = invstd * (dxh - sum1 / T(sc.d) - xh * sum2 / T(sc.d))

type LayerNormBwdWEnv*[T] = object
  ## Per-iteration env of the CPU layernorm backward dW kernel, one parameter element per iteration.
  dw*, g*, x*: ptr UncheckedArray[T]
  scalars*: LayerNormBwdScalars

proc layernormBwdWBody*[T: SomeFloat](c: int32, env: ptr LayerNormBwdWEnv[T]) {.cdecl.} =
  ## CPU port of the k_layernorm_bwd_w math, one parameter element per iteration.
  ## Each iteration recomputes the mean and var of every set it touches.
  let sc = env.scalars
  var acc = T(0)
  for r in 0 ..< sc.rows:
    let xbase = layerNormSetBase(sc.xShape, sc.xs, sc.xOff, r * sc.d)
    let gi = layerNormSetBase(sc.gShape, sc.gStrides, sc.gOff, r * sc.d + c)
    var mean = T(0)
    for j in 0 ..< sc.d:
      mean += env.x[][xbase + j * sc.xStep]
    mean /= T(sc.d)
    var vr = T(0)
    for j in 0 ..< sc.d:
      let dv = env.x[][xbase + j * sc.xStep] - mean
      vr += dv * dv
    vr /= T(sc.d)
    let xh = (env.x[][xbase + c * sc.xStep] - mean) / sqrt(vr + T(sc.eps))
    acc += env.g[][gi] * xh
  env.dw[][c] = acc

type LayerNormBwdBEnv*[T] = object
  ## Per-iteration env of the CPU layernorm backward dB kernel, one parameter element per iteration.
  db*, g*: ptr UncheckedArray[T]
  scalars*: LayerNormBwdScalars

proc layernormBwdBBody*[T: SomeFloat](c: int32, env: ptr LayerNormBwdBEnv[T]) {.cdecl.} =
  ## CPU port of the k_layernorm_bwd_b math, one parameter element per iteration.
  let sc = env.scalars
  var acc = T(0)
  for r in 0 ..< sc.rows:
    let i0 = r * sc.d + c
    let gi = layerNormSetBase(sc.gShape, sc.gStrides, sc.gOff, i0)
    acc += env.g[][gi]
  env.db[][c] = acc
