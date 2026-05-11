# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests Tensor lifetime management - adapted from bug_test_cpp_nim_destructors.nim

import
  workspace/libtorch/src/tensors,
  workspace/libtorch/libtorch_testutils

proc main() =
  # =============================================================================
  # Tensor basic lifetime
  # =============================================================================

  runTest "stack-to-stack copy":
    proc(): bool =
      let a = zeros(2, 3, kFloat32)
      let b = a
      doAssert b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3
      doAssert a.equal(b)
      true

  runTest "stack-to-stack move":
    proc(): bool =
      var a = zeros(2, 3, kFloat32)
      var b = move a
      doAssert b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3
      doAssert b.isDefined()
      true

  runTest "sink parameter":
    proc(): bool =
      proc takeTensor(t: sink Tensor): Tensor =
        t
      let a = zeros(2, 3, kFloat32)
      let b = takeTensor(a)
      doAssert b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3
      true

  runTest "seq[Tensor] add (copy)":
    proc(): bool =
      var s: seq[Tensor] = @[]
      let a = zeros(2, 3, kFloat32)
      s.add(a)
      doAssert s[0].dim() == 2 and s[0].size(0) == 2 and s[0].size(1) == 3
      true

  runTest "seq[Tensor] add (move)":
    proc(): bool =
      var s: seq[Tensor] = @[]
      var a = zeros(2, 3, kFloat32)
      s.add(move a)
      doAssert s[0].dim() == 2 and s[0].size(0) == 2 and s[0].size(1) == 3
      true

  runTest "seq[Tensor] indexed access after add":
    proc(): bool =
      var s: seq[Tensor] = @[]
      let a = zeros(2, 3, kFloat32)
      s.add(a)
      let b = s[0]
      doAssert b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3
      true

  runTest "seq[Tensor] multiple elements":
    proc(): bool =
      var s: seq[Tensor] = @[]
      for i in 0 ..< 10:
        s.add(full(3, float32(i), kFloat32))
      doAssert s.len == 10
      for i in 0 ..< 10:
        doAssert s[i][0].item(float32) == float32(i)
      true

  runTest "seq[Tensor] pre-alloc and assign":
    proc(): bool =
      var s: seq[Tensor] = newSeq[Tensor](3)
      s[0] = zeros(2, kFloat32)
      s[1] = ones(2, kFloat32)
      s[2] = zeros(2, kFloat32)
      doAssert s[0][0].item(float32) == 0.0
      doAssert s[1][0].item(float32) == 1.0
      true

  runTest "seq[Tensor] with grow-in-place (realloc)":
    proc(): bool =
      var s: seq[Tensor] = @[]
      for i in 0 ..< 100:
        s.add(full(1, float32(i), kFloat32))
      doAssert s.len == 100
      for i in 0 ..< 100:
        doAssert s[i][0].item(float32) == float32(i)
      true

  # =============================================================================
  # Tensor in embedded objects
  # =============================================================================

  runTest "object with Tensor field - stack copy":
    proc(): bool =
      type MyObj = object
        t: Tensor
        n: int
      let a = MyObj(t: zeros(2, kFloat32), n: 42)
      let b = a
      doAssert b.t.dim() == 1 and b.t.size(0) == 2
      doAssert b.n == 42
      true

  runTest "object with Tensor field - seq add (copy)":
    proc(): bool =
      type MyObj = object
        t: Tensor
        n: int
      var s: seq[MyObj] = @[]
      let a = MyObj(t: zeros(2, kFloat32), n: 42)
      s.add(a)
      doAssert s[0].t.dim() == 1 and s[0].t.size(0) == 2
      doAssert s[0].n == 42
      true

  runTest "object with Tensor field - seq add (move)":
    proc(): bool =
      type MyObj = object
        t: Tensor
        n: int
      var s: seq[MyObj] = @[]
      var a = MyObj(t: zeros(2, kFloat32), n: 42)
      s.add(move a)
      doAssert s[0].t.dim() == 1 and s[0].t.size(0) == 2
      doAssert s[0].n == 42
      true

  runTest "nested objects with Tensor":
    proc(): bool =
      type Inner = object
        t: Tensor
      type Outer = object
        inner: Inner
        n: int
      var s: seq[Outer] = @[]
      let o = Outer(inner: Inner(t: zeros(2, kFloat32)), n: 42)
      s.add(o)
      doAssert s[0].inner.t.dim() == 1 and s[0].inner.t.size(0) == 2
      true

  runTest "ptr[T] containing Tensor":
    proc(): bool =
      type MyObj = object
        t: Tensor
      let p = cast[ptr MyObj](alloc(sizeof(MyObj)))
      p.t = zeros(2, kFloat32)
      doAssert p.t.dim() == 1 and p.t.size(0) == 2
      dealloc(p)
      true

  runTest "ref object containing Tensor":
    proc(): bool =
      type MyRef = ref object
        t: Tensor
      var s: seq[MyRef] = @[]
      let r = MyRef(t: zeros(2, kFloat32))
      s.add(r)
      doAssert s[0].t.dim() == 1 and s[0].t.size(0) == 2
      true

  # =============================================================================
  # Tensor refcount correctness
  # =============================================================================

  runTest "multiple copies all valid":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = a
      let c = b
      let d = c
      doAssert a.isDefined()
      doAssert b.isDefined()
      doAssert c.isDefined()
      doAssert d.isDefined()
      doAssert a.equal(b)
      doAssert b.equal(c)
      doAssert c.equal(d)
      true

  runTest "scope exit of copies":
    proc(): bool =
      proc inner(): Tensor =
        ones(2, kFloat32)
      let result = inner()
      doAssert result.isDefined()
      doAssert result[0].item(float32) == 1.0
      true

  runTest "scope exit of seq[Tensor]":
    proc(): bool =
      proc inner(): seq[Tensor] =
        var s: seq[Tensor] = @[]
        s.add(ones(2, kFloat32))
        s.add(zeros(2, kFloat32))
        return s
      let result = inner()
      doAssert result.len == 2
      let t0 = result[0]
      let t1 = result[1]
      doAssert t0[0].item(float32) == 1.0
      doAssert t1[0].item(float32) == 0.0
      true

  runTest "seq element assignment after scope exit":
    proc(): bool =
      type Wrapper = object
        t: Tensor
      proc inner(): Wrapper =
        Wrapper(t: ones(2, kFloat32))
      var s: seq[Wrapper] = @[]
      s.add(inner())
      doAssert s[0].t[0].item(float32) == 1.0
      true

  # =============================================================================
  # Tensor clone (deep copy)
  # =============================================================================

  runTest "clone creates independent tensor":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = clone(a)
      let c = b + 1
      doAssert a[0, 0].item(float32) == 1.0
      doAssert c[0, 0].item(float32) == 2.0
      true

  runTest "clone vs copy share no memory":
    proc(): bool =
      let a = ones(2, 3, kFloat32)
      let b = clone(a)
      doAssert not a.is_alias_of(b)
      true


when isMainModule:
  main()
