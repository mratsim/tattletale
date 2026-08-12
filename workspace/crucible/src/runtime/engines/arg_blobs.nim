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

import std/macros
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

macro flattenBlobs*(args: untyped, storage: var seq[byte]): untyped =
  ## Flatten a tuple of typed kernel args into ArgBlobs, in tuple order.
  ## By-value scalars are memcpy'd into `storage` (pre-sized — no realloc, so
  ## the blobs' data pointers stay stable). A bare scalar (e.g. `(42'u32)` — a
  ## parenthesized expr, not a 1-tuple) is accepted as a single by-value blob.
  ##
  ## Implemented as a macro that emits `blobOf(el, storage)` for each original
  ## argument expression. A template that copies the tuple first (`var t =
  ## args`) is broken on this Nim 2.2.10 build: seq fields in a copied tuple
  ## go through `eqcopy` and lose their buffer identity (the blob data pointer
  ## no longer points at the caller's seq), silently corrupting the upload.
  ##
  ## The blobs reference the caller's buffers and `storage`: keep the args
  ## tuple live for the whole launch so the addresses stay valid until
  ## `runImpl` consumes them.
  let els =
    if args.kind in {nnkPar, nnkTupleConstr, nnkBracket}: args
    else: newTree(nnkPar, args)   # bare scalar/parenthesized expr
  let sizeVar = genSym(nskVar, "scalarBytes")
  let blobsVar = genSym(nskVar, "blobs")
  let prep = newStmtList()   # per-element scalar-size `when` statements
  let append = newStmtList() # per-element blob construction
  for el in els:
    prep.add quote do:
      when (`el` is seq) or (`el` is array) or (`el` is string):
        discard
      else:
        `sizeVar` += sizeof(`el`)
    append.add quote do:
      `blobsVar`.add blobOf(`el`, `storage`)
  result = quote do:
    block:
      `storage`.setLen(0)
      var `sizeVar` = 0
      `prep`
      `storage`.setLen(`sizeVar`)
      var `blobsVar`: seq[ArgBlob]
      `append`
      `blobsVar`
