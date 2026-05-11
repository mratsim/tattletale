#
#
#            Vec Container: Workaround for Nim C++ default(T) Bug
#        (c) Copyright 2025 Tattletale contributors
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#

## Vec: A fixed-size vector container for C++ types with destructors.
##
## This container avoids Nim's `default(T)` bug (generates `{}` in C++)
## by using explicit placement-new for construction and explicit destructor calls.
##
## Key features:
## - Fixed size after construction (no reallocation)
## - Proper destructor support for C++ types like `torch::Tensor`
## - Uses alignedAlloc/alignedDealloc from allocs.nim
## - Implements Nim's lifetime hooks: `=destroy`, `=copy`, `=sink`, `=wasMoved`, `=dup`

import
  std/[strformat, typetraits],
  ./lowlevel/allocs

proc placementNew[T](p: ptr T): ptr T {.importcpp: "(new (#) '*0(@))", nodecl, discardable.}
  ## Default-construct an object at the given memory location via placement-new.

const Alignment = 64

type
  Vec*[T] = object
    ## Manual memory-managed vector for C++ types with fixed size.
    ## Avoids Nim's default(T) bug by using explicit C++ construction.
    len: int
    data: ptr UncheckedArray[T]

func new*[T](_: type Vec[T], len: int): Vec[T] =
  ## Create Vec with exact length.
  result.len = len
  if len > 0:
    result.data = allocHeapArrayAligned(T, len, Alignment)
    # Explicitly default-construct each element via C++ ctor
    for i in 0..<len:
      placementNew(result.data[i].addr)
  else:
    result.data = nil

proc `=destroy`*[T](v: var Vec[T]) =
  ## Destroy all elements, free memory.
  if v.data != nil:
    when not supportsCopyMem(T):
      for i in 0..<v.len:
        `=destroy`(v.data[i])
    freeHeapAligned(v.data)
    v.data = nil
  v.len = 0

func `=wasMoved`*[T](v: var Vec[T]) {.inline.} =
  ## Mark as moved (set data to nil, len to 0).
  v.data = nil
  v.len = 0

func dupImpl[T](dst: var Vec[T], src: Vec[T]) {.nodestroy.} =
  ## Duplicate without destroying source.
  ## Optimized for return value optimization (RVO).
  dst.len = src.len
  if src.len > 0:
    dst.data = allocHeapArrayAligned(T, src.len, Alignment)
    when supportsCopyMem(T):
      copyMem(dst.data, src.data, src.len * sizeof(T))
    else:
      for i in 0..<src.len:
        placementNew(dst.data[i].addr)
        `=copy`(dst.data[i], src.data[i])
  else:
    dst.data = nil

func `=dup`*[T](src: Vec[T]): Vec[T] {.nodestroy, inline.} =
  result.dupImpl(src)

proc `=copy`*[T](dst: var Vec[T], src: Vec[T]) {.inline.} =
  ## Deep copy with proper lifetime management.
  ## Uses copyMem for trivially copyable types (supportsCopyMem).
  `=destroy`(dst)  # Destroy old elements first
  dst.dupImpl(src)

proc `=sink`*[T](dst: var Vec[T], src: Vec[T]) {.inline.} =
  ## Move semantics: steal pointer, zero source.
  `=destroy`(dst)
  dst.len = src.len
  dst.data = src.data
  # Note: compiler automatically calls =wasMoved on src after this proc returns

func len*[T](v: Vec[T]): int {.inline.} =
  ## Get current length.
  v.len

func `[]`*[T](v: Vec[T], i: Natural): lent T {.inline.} =
  ## Read access (returns lent to avoid copy).
  ## Bounds checking is enabled with compileOption("boundChecks").
  when compileOption("boundChecks"):
    if i >= v.len:
      raise newException(IndexError, "[ttt] Index '" & $i & "' out of bounds (length: " & $v.len & ")")
  v.data[i]

func `[]=`*[T](v: var Vec[T], i: Natural, item: sink T) {.inline.} =
  ## Write access (move assignment).
  ## Bounds checking is enabled with compileOption("boundChecks").
  when compileOption("boundChecks"):
    if i >= v.len:
      raise newException(IndexError, "[ttt] Index '" & $i & "' out of bounds (length: " & $v.len & ")")
  v.data[i] = item

template toOpenArray*[T](v: Vec[T]): openArray[T] =
  ## Convert to openArray for iteration.
  ## Returns a view into the Vec's data - do not modify Vec while iterating.
  ##
  ## TODO: cannot be made a converter due to
  ## https://github.com/nim-lang/Nim/issues/16848
  toOpenArray(v.data, 0, v.len - 1)

iterator items*[T](v: Vec[T]): lent T =
  ## Iterate over elements (read-only).
  for i in 0..<v.len:
    yield v.data[i]

iterator mitems*[T](v: var Vec[T]): var T =
  ## Iterate over elements (mutable).
  for i in 0..<v.len:
    yield v.data[i]

# #######################################################################
#
#               Syntactic sugar for Torch and Nim interop
#
# #######################################################################

import workspace/libtorch/src/raw/abi/c10

func asTorchView*[T](v: Vec[T]): ArrayRef[T] {.inline.} =
  ## Convert Vec to ArrayRef view.
  ## Returns a non-owning view - Vec must outlive the ArrayRef.
  if v.len == 0:
    init(ArrayRef[T])
  else:
    init(ArrayRef[T], v.data[0].addr, v.len)