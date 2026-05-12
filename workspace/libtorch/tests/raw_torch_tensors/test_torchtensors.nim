# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/sequtils
import std/complex
import std/sugar

import workspace/libtorch/src/raw_libtorch
import ./utils/[torch_tensors_overloads, raw_libtorch_testutils]

proc main() =
  # -----------------------------------------------------------------------
  # Operator precedence

  runTest "Operator precedence: + and *":
    proc(): bool =
      let a = [[1, 2], [3, 4]].toTorchTensor()
      let b = -a
      doAssert b * a + b == [[-2, -6], [-12, -20]].toTorchTensor()
      doAssert b * (a + b) == [[0, 0], [0, 0]].toTorchTensor()
      true

  runTest "Operator precedence: + and .abs":
    proc(): bool =
      let a = [[1, 2], [3, 4]].toTorchTensor()
      let b = -a
      doAssert a + b.abs == [[2, 4], [6, 8]].toTorchTensor()
      doAssert (a + b).abs == [[0, 0], [0, 0]].toTorchTensor()
      true

  # -----------------------------------------------------------------------
  # Tensor creation

  runTest "Tensor creation: eye":
    proc(): bool =
      let t = eye(2, kInt64)
      doAssert t == [[1, 0], [0, 1]].toTorchTensor()
      true

  runTest "Tensor creation: zeros":
    proc(): bool =
      let shape = [2, 3]
      let t = zeros(shape, kFloat32)
      doAssert t == [[0.0'f32, 0.0, 0.0], [0.0'f32, 0.0, 0.0]].toTorchTensor()
      true

  runTest "Tensor creation: linspace":
    proc(): bool =
      let steps = 120'i64
      let reft = toSeq(0..<120).map(x => float64(x)/float64(steps-1)).toTorchTensor()
      let t = linspace(0.0, 1.0, steps, kFloat64)
      let rel_error = mean(t - reft)
      doAssert rel_error.item(float64) <= 1e-12
      true

  runTest "Tensor creation: arange":
    proc(): bool =
      let steps = 130'i64
      let step = 1.0/float64(steps)
      let t = arange(0.0, 1.0, step, kFloat64)
      for i in 0..<130:
        let val = t[i].item(float64)
        let refval: float64 = i.float64 / 130.0
        doAssert (val - refval) < 1e-12
      true

  # -----------------------------------------------------------------------
  # Tensor utils

  runTest "Tensor utils: Print":
    proc(): bool =
      let shape = [2, 3, 4]
      let t = rand(shape, kfloat64)
      echo t
      true

  runTest "Tensor utils: sort, argsort":
    proc(): bool =
      let t = [2, 3, 4, 1, 5, 6].toTorchTensor()
      let
        s = t.sort()
        args = t.argsort()
      doAssert s.get(0) == [1, 2, 3, 4, 5, 6].toTorchTensor()
      doAssert s.get(1) == args
      doAssert args == [3, 0, 1, 2, 4, 5].toTorchTensor()
      true

  runTest "Tensor utils: all, any":
    proc(): bool =
      true

  runTest "Tensor utils: squeeze, unsqueeze":
    proc(): bool =
      true

  # -----------------------------------------------------------------------
  # Operations

  runTest "Operations: add, addmv, addmm":
    proc(): bool =
      true

  runTest "Operations: matmul, mm, bmm":
    proc(): bool =
      true

  # -----------------------------------------------------------------------
  # FFT1D

  runTest "FFT1D: item(Complex64)":
    proc(): bool =
      let shape = [8]
      let c64input = rand(shape, kComplexF64)
      let m: TorchComplex[float64] = c64input[0].item(Complex64)
      doAssert m.real is float64
      doAssert m.imag is float64
      true

  runTest "FFT1D: fft, ifft":
    proc(): bool =
      let shape = [8]
      let c64input = rand(shape, kComplexF64)
      let fftout = fft(c64input)
      let ifftout = ifft(fftout)
      let max_input = max(abs(ifftout)).item(float64)
      var rel_diff = abs(ifftout - c64input)
      rel_diff /= max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

  runTest "FFT1D: rfft, irfft":
    proc(): bool =
      let shape = [8]
      let f64input = rand(shape, kfloat64)
      let fftout = rfft(f64input)
      let ifftout = irfft(fftout)
      let max_input = max(abs(ifftout)).item(float64)
      var rel_diff = abs(ifftout - f64input)
      rel_diff /= max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

  # -----------------------------------------------------------------------
  # FFT2D

  runTest "FFT2D: fft2, ifft2":
    proc(): bool =
      let shape = [3, 5]
      let c64input = rand(shape, kComplexF64)
      let fft2out = fft2(c64input)
      let ifft2out = ifft2(fft2out)
      let max_input = max(abs(ifft2out)).item(float64)
      var rel_diff = abs(ifft2out - c64input)
      rel_diff /= max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

  # -----------------------------------------------------------------------
  # FFTND

  runTest "FFTND: fftn, ifftn":
    proc(): bool =
      let shape = [3, 4, 5]
      let c64input = rand(shape, kComplexF64)
      let fftnout = fftn(c64input)
      let ifftnout = ifftn(fftnout)
      let max_input = max(abs(ifftnout)).item(float64)
      var rel_diff = abs(ifftnout - c64input)
      rel_diff /= max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

when isMainModule:
  main()
