# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests TorchTensor lifetime management (Nim ↔ C++ `torch::Tensor` / `c10::intrusive_ptr`)
##
## Goal: Verify that TorchTensor can be safely copied, moved, and stored in sequences
## without double-free, use-after-free, or refcount leaks.
##
## PyTorch uses `c10::intrusive_ptr<TensorImpl>` internally. The C++ semantics are:
## - Copy constructor: increments refcount (shared ownership)
## - Move constructor: steals pointer, zeros source (no refcount change)
## - Destructor: decrements refcount (frees when refcount reaches 0)
## - `reset()`: releases internal pointer (decrements refcount)
##
## Nim mapping:
## - `=copy`: must increment refcount (use C++ copy ctor or clone)
## - `=sink`: can use raw memcpy (source will be zeroed by `=wasMoved`)
## - `=wasMoved`: must zero the tensor to prevent double-free (use `reset()`)
## - `=destroy`: must decrement refcount (use C++ destructor or `reset()`)

import
  std/unittest,
  workspace/libtorch as F,
  workspace/libtorch/src/torch_tensors_sugar

proc main() =
#   suite "TorchTensor basic lifetime":
#     test "stack-to-stack copy":
#       let a = F.zeros([2, 3], F.kFloat32)
#       let b = a
#       check b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3
#       check a == b

#     test "stack-to-stack move":
#       var a = F.zeros([2, 3], F.kFloat32)
#       var b = move a
#       check b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3
#       check b.isDefined()

#     test "sink parameter":
#       proc takeTensor(t: sink F.TorchTensor): F.TorchTensor =
#         t

#       let a = F.zeros([2, 3], F.kFloat32)
#       let b = takeTensor(a)
#       check b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3

#     test "seq[TorchTensor] add (copy)":
#       var seq: seq[F.TorchTensor] = @[]
#       let a = F.zeros([2, 3], F.kFloat32)
#       seq.add(a)
#       check seq[0].dim() == 2 and seq[0].size(0) == 2 and seq[0].size(1) == 3

#     test "seq[TorchTensor] add (move)":
#       var seq: seq[F.TorchTensor] = @[]
#       var a = F.zeros([2, 3], F.kFloat32)
#       seq.add(move a)
#       check seq[0].dim() == 2 and seq[0].size(0) == 2 and seq[0].size(1) == 3

#     test "seq[TorchTensor] indexed access after add":
#       var seq: seq[F.TorchTensor] = @[]
#       let a = F.zeros([2, 3], F.kFloat32)
#       seq.add(a)
#       let b = seq[0]
#       check b.dim() == 2 and b.size(0) == 2 and b.size(1) == 3

#     test "seq[TorchTensor] multiple elements":
#       var seq: seq[F.TorchTensor] = @[]
#       for i in 0 ..< 10:
#         seq.add(F.full([3], float32(i), F.kFloat32))
#       check seq.len == 10
#       for i in 0 ..< 10:
#         check seq[i].item(float32) == float32(i)

#     test "seq[TorchTensor] pre-alloc and assign":
#       var seq: seq[F.TorchTensor] = newSeq[F.TorchTensor](3)
#       seq[0] = F.zeros([2], F.kFloat32)
#       seq[1] = F.ones([2], F.kFloat32)
#       seq[2] = F.zeros([2], F.kFloat32)
#       check seq[0].item(float32) == 0.0
#       check seq[1].item(float32) == 1.0

#     test "seq[TorchTensor] with grow-in-place (realloc)":
#       var seq: seq[F.TorchTensor] = @[]
#       for i in 0 ..< 100:
#         seq.add(F.full([1], float32(i), F.kFloat32))
#       check seq.len == 100
#       # Verify all elements survived reallocs
#       for i in 0 ..< 100:
#         check seq[i].item(float32) == float32(i)

  suite "TorchTensor in embedded objects":

    test "object with TorchTensor field - stack copy":
      type MyObj = object
        t: F.TorchTensor
        n: int

      let a = MyObj(t: F.zeros([2], F.kFloat32), n: 42)
      let b = a
      check b.t.dim() == 1 and b.t.size(0) == 2
      check b.n == 42

  #   test "object with TorchTensor field - seq add (copy)":
  #     type MyObj = object
  #       t: F.TorchTensor
  #       n: int

  #     var seq: seq[MyObj] = @[]
  #     let a = MyObj(t: F.zeros([2], F.kFloat32), n: 42)
  #     seq.add(a)
  #     check seq[0].t.dim() == 1 and seq[0].t.size(0) == 2
  #     check seq[0].n == 42

  #   test "object with TorchTensor field - seq add (move)":
  #     type MyObj = object
  #       t: F.TorchTensor
  #       n: int

  #     var seq: seq[MyObj] = @[]
  #     var a = MyObj(t: F.zeros([2], F.kFloat32), n: 42)
  #     seq.add(move a)
  #     check seq[0].t.dim() == 1 and seq[0].t.size(0) == 2
  #     check seq[0].n == 42

  #   test "nested objects with TorchTensor":
  #     type Inner = object
  #       t: F.TorchTensor

  #     type Outer = object
  #       inner: Inner
  #       n: int

  #     var seq: seq[Outer] = @[]
  #     let o = Outer(inner: Inner(t: F.zeros([2], F.kFloat32)), n: 42)
  #     seq.add(o)
  #     check seq[0].inner.t.dim() == 1 and seq[0].inner.t.size(0) == 2

  #   test "ptr[T] containing TorchTensor":
  #     type MyObj = object
  #       t: F.TorchTensor

  #     let p = cast[ptr MyObj](alloc(sizeof(MyObj)))
  #     p.t = F.zeros([2], F.kFloat32)
  #     check p.t.dim() == 1 and p.t.size(0) == 2
  #     # Note: manual dealloc needed, no GC for raw ptr

  #   test "ref object containing TorchTensor":
  #     type MyRef = ref object
  #       t: F.TorchTensor

  #     var seq: seq[MyRef] = @[]
  #     let r = MyRef(t: F.zeros([2], F.kFloat32))
  #     seq.add(r)
  #     check seq[0].t.dim() == 1 and seq[0].t.size(0) == 2

  # suite "TorchTensor refcount correctness":

  #   test "multiple copies all valid":
  #     let a = F.ones([2, 3], F.kFloat32)
  #     let b = a
  #     let c = b
  #     let d = c
  #     # All copies should be independent references
  #     check a.isDefined()
  #     check b.isDefined()
  #     check c.isDefined()
  #     check d.isDefined()
  #     check a == b
  #     check b == c
  #     check c == d

  #   test "scope exit of copies":
  #     proc inner(): F.TorchTensor =
  #       let t = F.ones([2], F.kFloat32)
  #       return t

  #     let result = inner()
  #     check result.isDefined()
  #     check result.item(float32) == 1.0

  #   test "scope exit of seq[TorchTensor]":
  #     proc inner(): seq[F.TorchTensor] =
  #       var s: seq[F.TorchTensor] = @[]
  #       s.add(F.ones([2], F.kFloat32))
  #       s.add(F.zeros([2], F.kFloat32))
  #       return s

  #     let result = inner()
  #     check result.len == 2
  #     check result[0].item(float32) == 1.0
  #     check result[1].item(float32) == 0.0

  #   test "seq element assignment after scope exit":
  #     type Wrapper = object
  #       t: F.TorchTensor

  #     proc inner(): Wrapper =
  #       Wrapper(t: F.ones([2], F.kFloat32))

  #     var seq: seq[Wrapper] = @[]
  #     seq.add(inner())
  #     check seq[0].t.item(float32) == 1.0

  # suite "TorchTensor clone (deep copy)":

  #   test "clone creates independent tensor":
  #     let a = F.ones([2, 3], F.kFloat32)
  #     let b = a.clone()
  #     # Modifying b should not affect a
  #     let c = b.add(1.0)
  #     check a.item(float32) == 1.0
  #     check c.item(float32) == 2.0

  #   test "clone vs copy share no memory":
  #     let a = F.ones([2, 3], F.kFloat32)
  #     let b = a.clone()
  #     check not a.is_alias_of(b)

when isMainModule:
  main()
