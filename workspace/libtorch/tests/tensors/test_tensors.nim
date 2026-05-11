# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Core Tensor wrapper tests: creation, math, operators, reductions, FFT.

import
  std/sequtils,
  std/complex,
  std/sugar,
  workspace/libtorch/src/tensors,
  workspace/libtorch/libtorch_testutils

proc runTests*() =
  # -----------------------------------------------------------------------
  # Factory / creation

  runTest "zeros":
    proc(): bool =
      let t = zeros(2, 3, kFloat32)
      doAssert t.dim() == 2
      doAssert t.size(0) == 2
      doAssert t.size(1) == 3
      # item() only works on singleton tensors, use indexing for multi-dim
      doAssert t[0, 0].item(float32) == 0.0
      true

  runTest "ones":
    proc(): bool =
      let t = ones(2, 3, kFloat32)
      doAssert t[0, 0].item(float32) == 1.0
      true

  runTest "full":
    proc(): bool =
      let t = full(2, 3, 42.0'f32, kFloat32)
      doAssert t[0, 0].item(float32) == 42.0'f32
      true

  runTest "eye":
    proc(): bool =
      let t = eye(2, kInt64)
      doAssert t[0, 0].item(int64) == 1
      doAssert t[0, 1].item(int64) == 0
      doAssert t[1, 0].item(int64) == 0
      doAssert t[1, 1].item(int64) == 1
      true

  runTest "linspace":
    proc(): bool =
      let steps = 120'i64
      let t = linspace(0.0, 1.0, steps, kFloat64)
      let reft = toSeq(0..<120).map(x => float64(x) / float64(steps - 1)).toTensor()
      let rel_error = mean(t - reft)
      doAssert rel_error.item(float64) <= 1e-12
      true

  runTest "arange":
    proc(): bool =
      let steps = 130'i64
      let step = 1.0 / float64(steps)
      let t = arange(0.0, 1.0, step, kFloat64)
      for i in 0 ..< 130:
        let val = t[i].item(float64)
        let refval: float64 = float64(i) / 130.0
        doAssert (val - refval).abs <= 1e-12
      true

  runTest "from_blob":
    proc(): bool =
      var data: array[4, float32] = [1.0, 2.0, 3.0, 4.0]
      let t = from_blob(data[0].unsafeAddr, [2, 2], kFloat32)
      doAssert t[0, 0].item(float32) == 1.0
      doAssert t[1, 1].item(float32) == 4.0
      true

  runTest "clone":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = clone(a)
      doAssert not a.is_alias_of(b)
      true

  # -----------------------------------------------------------------------
  # Operator precedence

  runTest "+ and *":
    proc(): bool =
      let a = toTensor([[1, 2], [3, 4]]).to(kFloat64)
      let b = -a
      doAssert (b * a + b).equal(toTensor([[-2, -6], [-12, -20]]).to(kFloat64))
      doAssert (b * (a + b)).equal(zeros(2, 2).to(kFloat64))
      true

  runTest "+ and abs":
    proc(): bool =
      let a = toTensor([[1, 2], [3, 4]]).to(kFloat64)
      let b = -a
      doAssert (a + abs(b)).equal(toTensor([[2, 4], [6, 8]]).to(kFloat64))
      doAssert abs(a + b).equal(zeros(2, 2).to(kFloat64))
      true

  # -----------------------------------------------------------------------
  # Math unary

  runTest "exp / log":
    proc(): bool =
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = log(exp(a))
      doAssert a.allClose(b)
      true

  runTest "sin / cos":
    proc(): bool =
      let t = toTensor(@[0.0]).to(kFloat64)
      doAssert sin(t)[0].item(float64) == 0.0
      doAssert cos(t)[0].item(float64) == 1.0
      true

  runTest "sqrt":
    proc(): bool =
      let t = toTensor(@[4.0, 9.0, 16.0]).to(kFloat64)
      doAssert sqrt(t)[0].item(float64) == 2.0
      doAssert sqrt(t)[1].item(float64) == 3.0
      true

  # -----------------------------------------------------------------------
  # Binary / Linear Algebra

  runTest "add":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = ones(2, 3, kFloat32) * 2.0
      let c = add(a, b)
      doAssert c[0, 0].item(float32) == 3.0
      true

  runTest "mm":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = ones(3, 4, kFloat32)
      let c = mm(a, b)
      doAssert c.dim() == 2
      doAssert c.size(0) == 2
      doAssert c.size(1) == 4
      doAssert c[0, 0].item(float32) == 3.0  # each row of a * each col of b = 3
      true

  runTest "matmul":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = ones(3, 4, kFloat32)
      let c = matmul(a, b)
      doAssert c.size(0) == 2
      doAssert c.size(1) == 4
      true

  runTest "dot":
    proc(): bool =
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = toTensor(@[4.0, 5.0, 6.0]).to(kFloat64)
      let d = dot(a, b)
      doAssert d.item(float64) == 32.0  # 1*4 + 2*5 + 3*6
      true

  # -----------------------------------------------------------------------
  # Comparison

  runTest "equal (bool)":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = ones(2, 3, kFloat32)
      let c = zeros(2, 3, kFloat32)
      doAssert equal(a, b)
      doAssert not equal(a, c)
      true

  runTest "eq (tensor)":
    proc(): bool =
      let a = toTensor(@[1.0, 2.0]).to(kFloat64)
      let b = toTensor(@[1.0, 3.0]).to(kFloat64)
      let eq_result = eq(a, b)
      doAssert eq_result[0].item(float64) == 1.0
      doAssert eq_result[1].item(float64) == 0.0
      true

  runTest "element-wise <. >. <=. >=. !=.":
    proc(): bool =
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = toTensor(@[1.0, 1.0, 4.0]).to(kFloat64)
      doAssert (a <. b)[1].item(float64) == 0.0  # 2 < 1 -> false
      doAssert (a >. b)[2].item(float64) == 0.0  # 3 > 4 -> false
      doAssert (a <=. b)[0].item(float64) == 1.0 # 1 <= 1 -> true
      doAssert (a >=. b)[0].item(float64) == 1.0 # 1 >= 1 -> true
      doAssert (a !=. b)[1].item(float64) == 1.0 # 2 != 1 -> true
      true

  runTest "allClose":
    proc(): bool =
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = toTensor(@[1.00001, 2.00001, 3.00001]).to(kFloat64)
      doAssert allClose(a, b)
      true

  # -----------------------------------------------------------------------
  # Reductions

  runTest "sum":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      doAssert sum(a).item(float32) == 6.0
      true

  runTest "mean":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      doAssert mean(a).item(float32) == 1.0
      true

  runTest "max":
    proc(): bool =
      let a = toTensor([[1, 3], [2, 4]]).to(kFloat64)
      doAssert max(a).item(float64) == 4.0
      true

  runTest "min":
    proc(): bool =
      let a = toTensor([[1, 3], [2, 4]]).to(kFloat64)
      doAssert min(a).item(float64) == 1.0
      true

  runTest "argmax":
    proc(): bool =
      let a = toTensor(@[1.0, 5.0, 3.0]).to(kFloat64)
      doAssert argmax(a).item(int64) == 1
      true

  runTest "sum with axis":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let s = sum(a, axis = 1)
      doAssert s.size(0) == 2
      doAssert s[0].item(float32) == 3.0
      true

  runTest "min with axis (tuple)":
    proc(): bool =
      let a = toTensor([[1, 3], [2, 4]]).to(kFloat64)
      let (vals, idx) = min(a, axis = 1)
      doAssert vals[0].item(float64) == 1.0
      doAssert vals[1].item(float64) == 2.0
      true

  runTest "sort":
    proc(): bool =
      let t = toTensor(@[2, 3, 4, 1, 5, 6]).to(kInt64)
      let (s, args) = sort(t)
      doAssert s[0].item(int64) == 1
      doAssert s[1].item(int64) == 2
      doAssert args[0].item(int64) == 3  # index of 1
      doAssert args[1].item(int64) == 0  # index of 2
      true

  # -----------------------------------------------------------------------
  # Shape manipulation

  runTest "reshape":
    proc(): bool =
      let a = arange(12, kFloat64)
      let b = reshape(a, 3, 4)
      doAssert b.size(0) == 3
      doAssert b.size(1) == 4
      true

  runTest "view":
    proc(): bool =
      let a = arange(12, kFloat64)
      let b = view(a, 3, 4)
      doAssert b.size(0) == 3
      doAssert b.size(1) == 4
      true

  runTest "permute":
    proc(): bool =
      let a = zeros(2, 3, 4, kFloat32)
      let b = permute(a, 2, 1, 0)
      doAssert b.size(0) == 4
      doAssert b.size(1) == 3
      doAssert b.size(2) == 2
      true

  runTest "transpose":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = transpose(a, 0, 1)
      doAssert b.size(0) == 3
      doAssert b.size(1) == 2
      true

  runTest "t":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = t(a)
      doAssert b.size(0) == 3
      doAssert b.size(1) == 2
      true

  runTest "squeeze / unsqueeze":
    proc(): bool =
      let a = ones(1, 3, 1, kFloat32)
      let b = squeeze(a)
      doAssert b.size(0) == 3
      let c = unsqueeze(b, 0)
      doAssert c.dim() == 2
      doAssert c.size(0) == 1
      doAssert c.size(1) == 3
      true

  # -----------------------------------------------------------------------
  # Arithmetic operators

  runTest "+":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = ones(2, 3, kFloat32)
      let c = a + b
      doAssert c[0, 0].item(float32) == 2.0
      true

  runTest "-":
    proc(): bool =
      let a = full(2, 3, 5.0'f32, kFloat32)
      let b = ones(2, 3, kFloat32)
      let c = a - b
      doAssert c[0, 0].item(float32) == 4.0
      true

  runTest "*":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = full(2, 3, 3.0'f32, kFloat32)
      let c = a * b
      doAssert c[0, 0].item(float32) == 3.0
      true

  runTest "/":
    proc(): bool =
      let a = full(2, 3, 6.0'f32, kFloat32)
      let b = full(2, 3, 2.0'f32, kFloat32)
      let c = a / b
      doAssert c[0, 0].item(float32) == 3.0
      true

  runTest "scalar mixed":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = a + 2.0
      doAssert b[0, 0].item(float32) == 3.0
      let c = 2.0 * a
      doAssert c[0, 0].item(float32) == 2.0
      true

  runTest "in-place +=":
    proc(): bool =
      var a = ones(2, 3, kFloat32)
      a += ones(2, 3, kFloat32)
      doAssert a[0, 0].item(float32) == 2.0
      true

  runTest "in-place scalar *=":
    proc(): bool =
      var a = ones(2, 3, kFloat32)
      a *= 5.0
      doAssert a[0, 0].item(float32) == 5.0
      true

  # -----------------------------------------------------------------------
  # FFT

  runTest "fft / ifft":
    proc(): bool =
      let shape = @[8]
      let c64input = randn(shape, kComplexF64)
      let fftout = fft(c64input)
      let ifftout = ifft(fftout)
      let max_input = max(abs(ifftout)).item(float64)
      var rel_diff = abs(ifftout - c64input)
      rel_diff = rel_diff / max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

  runTest "rfft / irfft":
    proc(): bool =
      let shape = @[8]
      let f64input = randn(shape, kFloat64)
      let fftout = rfft(f64input)
      let ifftout = irfft(fftout)
      let max_input = max(abs(ifftout)).item(float64)
      var rel_diff = abs(ifftout - f64input)
      rel_diff = rel_diff / max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

  runTest "fft2 / ifft2":
    proc(): bool =
      let shape = @[3, 5]
      let c64input = randn(shape, kComplexF64)
      let fft2out = fft2(c64input)
      let ifft2out = ifft2(fft2out)
      let max_input = max(abs(ifft2out)).item(float64)
      var rel_diff = abs(ifft2out - c64input)
      rel_diff = rel_diff / max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

  runTest "fftn / ifftn":
    proc(): bool =
      let shape = @[3, 4, 5]
      let c64input = randn(shape, kComplexF64)
      let fftnout = fftn(c64input)
      let ifftnout = ifftn(fftnout)
      let max_input = max(abs(ifftnout)).item(float64)
      var rel_diff = abs(ifftnout - c64input)
      rel_diff = rel_diff / max_input
      doAssert mean(rel_diff).item(float64) < 1e-12
      true

when isMainModule:
  runTests()