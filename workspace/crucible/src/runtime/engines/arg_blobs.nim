# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Internal arg-flattening machinery — NOT part of the public HwEngine API.
##
## Lives outside `runtime/engines` (the public module) so the public surface
## stays limited to the HwEngine concept; the engine modules and their `run`
## templates import this module for blob construction.
##
## Containers (seq/array/string) → `(addr, len·sizeof(T))` device buffers;
## scalars (incl. raw pointer values) → `(addr, -sizeof(T))` by-value.
## `bool` is the exception: it marshals as a 4-byte i32 (see `blobOf`).

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  ArgBlob* = tuple[data: pointer, size: int]
    ## Type-erased internal layer:
    ##   size >= 0 → device buffer: memcpy `size` bytes host→device, bind as
    ##               buffer/SSBO/storage
    ##   size <  0 → trivial by-value scalar of `-size` bytes (no device alloc)
    ## The output of `run` is always treated as a device buffer (uploaded
    ## before launch, read back after) regardless of the sign of its size.

func blobOf*[T](x: seq[T], storage: var seq[byte]): ArgBlob {.inline.} =
  # A seq param is a shallow copy sharing the caller's refcounted buffer,
  # so addr x[0] stays valid for the whole launch (the args tuple holds
  # the caller's seq alive).
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil),
          size: x.len * sizeof(T))

template blobOf*[N, T](x: array[N, T], storage: var seq[byte]): ArgBlob =
  # Must stay a template: a by-value array param copies to the callee
  # stack and the blob would dangle after return, while literal-array
  # args (e.g. `[1'u32]`) are rvalues a `var` param cannot accept.
  (data: cast[pointer](addr x[0]), size: sizeof(x))

func blobOf*(x: string, storage: var seq[byte]): ArgBlob {.inline.} =
  # Same refcounted-buffer reasoning as the seq overload.
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil), size: x.len)

func blobOf*[T](x: T, storage: var seq[byte]): ArgBlob {.inline.} =
  let off = storage.len
  when T is bool:
    # Bool marshals as a 4-byte i32, the width every shader backend declares:
    # WGSL cannot use bool storage variables and emits i32, OpenCL C has no
    # bool and emits int. The value is 0 or 1, so the i32 read is exact and
    # CUDA/GLSL 1-byte `bool` kernels still see the right value in byte 0.
    storage.setLen(off + 4)
    var tmp = int32(x)
    copyMem(addr storage[off], addr tmp, 4)
    (data: cast[pointer](addr storage[off]), size: -4)
  else:
    storage.setLen(off + sizeof(T))
    var tmp = x   # make literals/consts addressable
    copyMem(addr storage[off], addr tmp, sizeof(T))
    (data: cast[pointer](addr storage[off]), size: -sizeof(T))

func outBlob*[T](x: var seq[T]): ArgBlob {.inline.} =
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil),
          size: x.len * sizeof(T))

func outBlob*[N, T](x: var array[N, T]): ArgBlob {.inline.} =
  (data: cast[pointer](addr x[0]), size: sizeof(x))

func outBlob*(x: var string): ArgBlob {.inline.} =
  (data: (if x.len > 0: cast[pointer](addr x[0]) else: nil), size: x.len)

func outBlob*[T](x: var T): ArgBlob {.inline.} =
  (data: cast[pointer](addr x), size: sizeof(T))

