# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Core Tensor wrapper tests: creation, math, operators, reductions, FFT.

import
  std/unittest,
  std/sequtils,
  std/complex,
  std/sugar,
  workspace/libtorch/src/tensors

proc runTests*() =
  # -----------------------------------------------------------------------
  # Factory / creation

  suite "Tensor creation":
    test "zeros":
      let t = zeros(2, 3, kFloat32)
      check t.dim() == 2
      check t.size(0) == 2
      check t.size(1) == 3
      check t.item(float32) == 0.0

    test "ones":
      let t = ones(2, 3, kFloat32)
      check t.item(float32) == 1.0

    test "full":
      let t = full(2, 3, 42.0'f32, kFloat32)
      check t.item(float32) == 42.0'f32

    test "eye":
      let t = eye(2, kInt64)
      check t[0, 0].item(int64) == 1
      check t[0, 1].item(int64) == 0
      check t[1, 0].item(int64) == 0
      check t[1, 1].item(int64) == 1

    test "linspace":
      let steps = 120'i64
      let t = linspace(0.0, 1.0, steps, kFloat64)
      let reft = toSeq(0..<120).map(x => float64(x) / float64(steps - 1)).toTensor()
      let rel_error = mean(t - reft)
      check rel_error.item(float64) <= 1e-12

    test "arange":
      let steps = 130'i64
      let step = 1.0 / float64(steps)
      let t = arange(0.0, 1.0, step, kFloat64)
      for i in 0 ..< 130:
        let val = t[i].item(float64)
        let refval: float64 = float64(i) / 130.0
        check (val - refval).abs <= 1e-12

    test "from_blob":
      var data: array[4, float32] = [1.0, 2.0, 3.0, 4.0]
      let t = from_blob(data[0].unsafeAddr, 2, 2, kFloat32)
      check t[0, 0].item(float32) == 1.0
      check t[1, 1].item(float32) == 4.0

    test "clone":
      let a = ones(2, 3, kFloat32)
      let b = clone(a)
      check not a.is_alias_of(b)

  # -----------------------------------------------------------------------
  # Operator precedence

  suite "Operator precedence":
    test "+ and *":
      let a = toTensor([[1, 2], [3, 4]]).to(kFloat64)
      let b = -a
      check (b * a + b).equal(toTensor([[-2, -6], [-12, -20]]).to(kFloat64))
      check (b * (a + b)).equal(zeros(2, 2).to(kFloat64))

    test "+ and abs":
      let a = toTensor([[1, 2], [3, 4]]).to(kFloat64)
      let b = -a
      check (a + abs(b)).equal(toTensor([[2, 4], [6, 8]]).to(kFloat64))
      check abs(a + b).equal(zeros(2, 2).to(kFloat64))

  # -----------------------------------------------------------------------
  # Math unary

  suite "Math unary":
    test "exp / log":
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = log(exp(a))
      check a.allClose(b)

    test "sin / cos":
      let t = toTensor(@[0.0]).to(kFloat64)
      check sin(t)[0].item(float64) == 0.0
      check cos(t)[0].item(float64) == 1.0

    test "sqrt":
      let t = toTensor(@[4.0, 9.0, 16.0]).to(kFloat64)
      check sqrt(t)[0].item(float64) == 2.0
      check sqrt(t)[1].item(float64) == 3.0

  # -----------------------------------------------------------------------
  # Binary / Linear Algebra

  suite "Linear Algebra":
    test "add":
      let a = ones(2, 3, kFloat32)
      let b = ones(2, 3, kFloat32) * 2.0
      let c = add(a, b)
      check c[0, 0].item(float32) == 3.0

    test "mm":
      let a = ones(2, 3, kFloat32)
      let b = ones(3, 4, kFloat32)
      let c = mm(a, b)
      check c.dim() == 2
      check c.size(0) == 2
      check c.size(1) == 4
      check c[0, 0].item(float32) == 3.0  # each row of a * each col of b = 3

    test "matmul":
      let a = ones(2, 3, kFloat32)
      let b = ones(3, 4, kFloat32)
      let c = matmul(a, b)
      check c.size(0) == 2
      check c.size(1) == 4

    test "dot":
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = toTensor(@[4.0, 5.0, 6.0]).to(kFloat64)
      let d = dot(a, b)
      check d.item(float64) == 32.0  # 1*4 + 2*5 + 3*6

  # -----------------------------------------------------------------------
  # Comparison

  suite "Comparison":
    test "equal (bool)":
      let a = ones(2, 3, kFloat32)
      let b = ones(2, 3, kFloat32)
      let c = zeros(2, 3, kFloat32)
      check equal(a, b)
      check not equal(a, c)

    test "eq (tensor)":
      let a = toTensor(@[1.0, 2.0]).to(kFloat64)
      let b = toTensor(@[1.0, 3.0]).to(kFloat64)
      let eq_result = eq(a, b)
      check eq_result[0].item(float64) == 1.0
      check eq_result[1].item(float64) == 0.0

    test "element-wise <. >. <=. >=. !=.":
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = toTensor(@[1.0, 1.0, 4.0]).to(kFloat64)
      check (a <. b)[1].item(float64) == 0.0  # 2 < 1 -> false
      check (a >. b)[2].item(float64) == 0.0  # 3 > 4 -> false
      check (a <=. b)[0].item(float64) == 1.0 # 1 <= 1 -> true
      check (a >=. b)[0].item(float64) == 1.0 # 1 >= 1 -> true
      check (a !=. b)[1].item(float64) == 1.0 # 2 != 1 -> true

    test "allClose":
      let a = toTensor(@[1.0, 2.0, 3.0]).to(kFloat64)
      let b = toTensor(@[1.00001, 2.00001, 3.00001]).to(kFloat64)
      check allClose(a, b)

  # -----------------------------------------------------------------------
  # Reductions

  suite "Reductions":
    test "sum":
      let a = ones(2, 3, kFloat32)
      check sum(a).item(float32) == 6.0

    test "mean":
      let a = ones(2, 3, kFloat32)
      check mean(a).item(float32) == 1.0

    test "max":
      let a = toTensor([[1, 3], [2, 4]]).to(kFloat64)
      check max(a).item(float64) == 4.0

    test "min":
      let a = toTensor([[1, 3], [2, 4]]).to(kFloat64)
      check min(a).item(float64) == 1.0

    test "argmax":
      let a = toTensor(@[1.0, 5.0, 3.0]).to(kFloat64)
      check argmax(a).item(int64) == 1

    test "sum with axis":
      let a = ones(2, 3, kFloat32)
      let s = sum(a, axis = 1)
      check s.size(0) == 2
      check s[0].item(float32) == 3.0

    test "min with axis (tuple)":
      let a = toTensor([[1, 3], [2, 4]]).to(kFloat64)
      let (vals, idx) = min(a, axis = 1)
      check vals[0].item(float64) == 1.0
      check vals[1].item(float64) == 2.0

    test "sort":
      let t = toTensor(@[2, 3, 4, 1, 5, 6]).to(kInt64)
      let (s, args) = sort(t)
      check s[0].item(int64) == 1
      check s[1].item(int64) == 2
      check args[0].item(int64) == 3  # index of 1
      check args[1].item(int64) == 0  # index of 2

  # -----------------------------------------------------------------------
  # Shape manipulation

  suite "Shape manipulation":
    test "reshape":
      let a = arange(12, kFloat64)
      let b = reshape(a, 3, 4)
      check b.size(0) == 3
      check b.size(1) == 4

    test "view":
      let a = arange(12, kFloat64)
      let b = view(a, 3, 4)
      check b.size(0) == 3
      check b.size(1) == 4

    test "permute":
      let a = zeros(2, 3, 4, kFloat32)
      let b = permute(a, 2, 1, 0)
      check b.size(0) == 4
      check b.size(1) == 3
      check b.size(2) == 2

    test "transpose":
      let a = ones(2, 3, kFloat32)
      let b = transpose(a, 0, 1)
      check b.size(0) == 3
      check b.size(1) == 2

    test "t":
      let a = ones(2, 3, kFloat32)
      let b = t(a)
      check b.size(0) == 3
      check b.size(1) == 2

    test "squeeze / unsqueeze":
      let a = ones(1, 3, 1, kFloat32)
      let b = squeeze(a)
      check b.size(0) == 3
      let c = unsqueeze(b, 0)
      check c.dim() == 2
      check c.size(0) == 1
      check c.size(1) == 3

  # -----------------------------------------------------------------------
  # Arithmetic operators

  suite "Arithmetic operators":
    test "+":
      let a = ones(2, 3, kFloat32)
      let b = ones(2, 3, kFloat32)
      let c = a + b
      check c[0, 0].item(float32) == 2.0

    test "-":
      let a = full(2, 3, 5.0'f32, kFloat32)
      let b = ones(2, 3, kFloat32)
      let c = a - b
      check c[0, 0].item(float32) == 4.0

    test "*":
      let a = ones(2, 3, kFloat32)
      let b = full(2, 3, 3.0'f32, kFloat32)
      let c = a * b
      check c[0, 0].item(float32) == 3.0

    test "/":
      let a = full(2, 3, 6.0'f32, kFloat32)
      let b = full(2, 3, 2.0'f32, kFloat32)
      let c = a / b
      check c[0, 0].item(float32) == 3.0

    test "scalar mixed":
      let a = ones(2, 3, kFloat32)
      let b = a + 2.0
      check b[0, 0].item(float32) == 3.0
      let c = 2.0 * a
      check c[0, 0].item(float32) == 2.0

    test "in-place +=":
      var a = ones(2, 3, kFloat32)
      a += ones(2, 3, kFloat32)
      check a[0, 0].item(float32) == 2.0

    test "in-place scalar *=":
      var a = ones(2, 3, kFloat32)
      a *= 5.0
      check a[0, 0].item(float32) == 5.0

  # -----------------------------------------------------------------------
  # FFT

  suite "FFT 1D":
    test "fft / ifft":
      let shape = @[8]
      let c64input = randn(shape, kComplexF64)
      let fftout = fft(c64input)
      let ifftout = ifft(fftout)
      let max_input = max(abs(ifftout)).item(float64)
      var rel_diff = abs(ifftout - c64input)
      rel_diff = rel_diff / max_input
      check mean(rel_diff).item(float64) < 1e-12

    test "rfft / irfft":
      let shape = @[8]
      let f64input = randn(shape, kFloat64)
      let fftout = rfft(f64input)
      let ifftout = irfft(fftout)
      let max_input = max(abs(ifftout)).item(float64)
      var rel_diff = abs(ifftout - f64input)
      rel_diff = rel_diff / max_input
      check mean(rel_diff).item(float64) < 1e-12

  suite "FFT 2D":
    test "fft2 / ifft2":
      let shape = @[3, 5]
      let c64input = randn(shape, kComplexF64)
      let fft2out = fft2(c64input)
      let ifft2out = ifft2(fft2out)

  suite "FFT ND":
    test "fftn / ifftn":
      let shape = @[3, 4, 5]
      let c64input = randn(shape, kComplexF64)
      let fftnout = fftn(c64input)
      let ifftnout = ifftn(fftnout)
      let max_input = max(abs(ifftnout)).item(float64)
      var rel_diff = abs(ifftnout - c64input)
      rel_diff = rel_diff / max_input
      check mean(rel_diff).item(float64) < 1e-12

when isMainModule:
  runTests()
