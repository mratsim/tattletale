# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/memfiles,
  std/tables,
  std/importutils,
  workspace/libtorch

import ./safetensors {.all.}
import ./collections {.all.}
privateAccess(SafetensorObj)
privateAccess(SafetensorsCollectionObj)

# #######################################################################
#
#               Safetensors + libtorch syntactic sugar
#
# #######################################################################
#
# The API here might change with the following consideration
# - How to allow fast loading (async Streams, parallel workers, direct to GPU, ...)
# - How to associate lifetimes of `MemFile` and `MemSlice`
#
#   Unfortunately MemFile predates `lent` and `openarray` as values view `{.experimental: "views".}`
#   so we don't get compiler-enforced borrow-checking.
#   https://github.com/nim-lang/nimony/issues/1517#issuecomment-3859350630
#
#   And this is not available yet
#   https://nim-lang.org/docs/manual.html#var-return-type-future-directions
#   `proc foo(other: Y; container: var X): var T from container`
#
# The borrow check for `var T` return types:
#   https://nim-lang.org/docs/manual.html#procedures-var-return-type
#
# is not applicable here because we allocate a fresh address to store MemSlice
# instead of using the input Safetensor or one of its field.

proc toTorchType*(dtype: ST_dtype): ScalarKind {.inline.} =
  ## Convert safetensors dtype to libtorch ScalarKind.
  ## Raises ValueError if no direct mapping exists.

  case dtype:
    of BOOL: kBool
    of U8:   kUint8
    of I8:   kInt8
    of I16:  kInt16
    of F16:  kFloat16
    of BF16: kBfloat16
    of I32:  kInt32
    of F32:  kFloat32
    of C64:  kComplexF64
    of F64:  kFloat64
    of I64:  kInt64
    else:
      raise newException(ValueError, "No direct libtorch mapping for safetensors dtype: " & $dtype)

proc hasTensor*(st: Safetensor, tensorName: string): bool {.inline.} =
  ## True when `st`'s header holds `tensorName`.
  ## Loaders use it to probe optional tensors (.bias, .mcg, .mul1).
  st.tensors.hasKey(tensorName)

proc hasTensor*(view: SafetensorsCollection, tensorName: string): bool {.inline.} =
  ## True when the collection's weight map routes `tensorName` to a file.
  view.weightMap.hasKey(tensorName)

when defined(TTT_ALLOC_TRACE):
  # Diagnostic allocation trace, compile-time gated, zero overhead and
  # zero output when the define is absent. Per getTensorOwned call one
  # flushed line names the callsite outside this file and the byte
  # delta, the log survives a mid-load death by design.
  import std/os
  import std/strformat
  import std/strutils

  var tttTraceRetainedBytes: int64 = 0
  var tttTraceSink: File = stderr
  var tttTraceSinkReady = false

  proc tttTraceFrame(line: string): tuple[ok: bool, file: string,
      num: int, procName: string] =
    ## Parse one Nim stack frame of the form file(num) procname or
    ## file(num, col) procname.
    let open = line.find('(')
    if open <= 0:
      return (false, "", 0, "")
    let close = line.find(')', open)
    if close < 0 or close + 2 > line.len:
      return (false, "", 0, "")
    let file = line[0 ..< open]
    if not file.endsWith(".nim"):
      return (false, "", 0, "")
    let body = line[open + 1 ..< close]
    let sep = body.find(',')
    let numText = if sep < 0: body else: body[0 ..< sep]
    var num = 0
    try:
      num = parseInt(numText)
    except ValueError:
      return (false, "", 0, "")
    let procName = line[close + 2 .. ^1]
    result = (true, file, num, procName)

  proc tttTraceLocate(frameLines: seq[string]): string =
    ## Innermost frame outside safetensors_libtorch.nim is the calling
    ## loader, when the next frame toward the copy machinery sits in
    ## libtorch it rides the line too. Nim tracebacks list outermost
    ## frames first, the scan runs from the innermost end.
    var parsed: seq[tuple[ok: bool, file: string, num: int, procName: string]]
    for line in frameLines:
      let f = tttTraceFrame(line)
      if f.ok:
        parsed.add f
    var idx = -1
    var i = parsed.len - 1
    while i >= 0:
      if not parsed[i].file.endsWith("safetensors_libtorch.nim"):
        idx = i
        break
      dec i
    if idx < 0:
      return "<unknown-callsite>"
    let c = parsed[idx]
    result = &"{c.file}:{c.num}:{c.procName}"
    if idx > 0 and "libtorch" in parsed[idx - 1].file:
      let u = parsed[idx - 1]
      result.add &" up={u.file}:{u.num}:{u.procName}"

  proc tttAllocTraceEmit(tensorName: string, view: Tensor) {.noinline.} =
    ## One flushed line per owned copy, delta counted before the copy
    ## sits so the last line of a dead run names the killing callsite.
    let delta = view.numel.int64 * view.itemsize.int64
    tttTraceRetainedBytes += delta
    if not tttTraceSinkReady:
      tttTraceSinkReady = true
      when defined(TTT_ALLOC_TRACE_LOG):
        let opened = open(TTT_ALLOC_TRACE_LOG, fmWrite)
        if opened.isNil:
          tttTraceSink = stderr
        else:
          tttTraceSink = opened
    let stackLines = getStackTrace().splitLines()
    let loc = tttTraceLocate(stackLines)
    tttTraceSink.writeLine(
      &"{loc} tensor={tensorName} delta={delta} retained={tttTraceRetainedBytes}")
    tttTraceSink.flushFile()

proc getTensorView*(st: Safetensor, tensorName: string): Tensor =
  ## Get a memory view to the tensor data.
  ## Returns a `Tensor` that views the underlying memory-mapped data.
  ##
  ## Memory safety:
  ##   ⚠️ WARNING: The returned `Tensor` is a view into `st.memFile`.
  ##   The tensor MUST NOT outlive the underlying memory mapping.
  ##   If the mapping is released, accessing this tensor will cause undefined behavior / crash.
  ##
  ## For safe tensor loading, use `getTensorOwned` instead.
  ## This is intended to:
  ## - build high-performance loading primitives for example, direct-to-GPU weight loading.
  ## - memory-mapped inference on CPU
  ##
  ## Lifetime:
  ##   The tensor is valid as long as `st` is valid, which is tied to
  ##   the `MemFile` acquired by `open` and released by the destructor.
  let view = st.getMmapView(tensorName)
  let info = st.tensors[tensorName]
  return from_blob(view.data, info.shape, info.dtype.toTorchType())

proc getTensorOwned*(st: Safetensor, tensorName: string, device = kCPU): Tensor =
  ## Get an owned copy of the tensor data.
  ## Returns a `TorchTensor` that owns its data, safe to use after the mapping is released.
  ##
  ## This is the recommended way to load tensors for inference.
  ## The tensor is cloned to `device` memory (default CPU).
  ##
  ## Args:
  ##   st: A loaded Safetensor (must remain valid during the copy)
  ##   tensorName: Name of the tensor to load
  ##
  ## Returns:
  ##   An owned `TorchTensor` on `device`.
  let view = st.getTensorView(tensorName)
  when defined(TTT_ALLOC_TRACE):
    tttAllocTraceEmit(tensorName, view)
  result = view.to(device, copy=true) # Force copy

proc getTensorOwned*(view: SafetensorsCollection, tensorName: string, device = kCPU): Tensor =
  ## Get an owned copy of the tensor data of `tensorName`.
  ## Weight map routes the name to the owning safetensor file.
  ## That file answers the header: source of truth for the data it holds.
  ##
  ## This is the recommended way to load tensors for inference.
  ## The tensor is cloned to `device` memory (default CPU).
  ##
  ## Args:
  ##   view: A collection that is alive for the duration of the copy
  ##   tensorName: Name of the tensor to load
  ##
  ## Returns:
  ##   An owned `TorchTensor` on `device`, independent of every mapping.
  ## Raises ValueError naming the tensor when the collection holds no entry.
  ## Trace coverage comes from the per-tensor proc above: the
  ## collection-level form routes through it so every call is counted
  ## exactly once.
  view.files[view.fileOf(tensorName)].getTensorOwned(tensorName, device)

proc getTensorView*(view: SafetensorsCollection, tensorName: string): Tensor =
  ## Zero-copy mmap view of the tensor data of `tensorName`,
  ## collection-level form of the per-file `getTensorView`: the weight
  ## map routes the name to the owning safetensor file and that file serves
  ## the view from its mapping, no owned copy is made.
  ##
  ## Lifetime: the returned `Tensor` borrows the mapping. The collection
  ## must outlive every tensor served this way, the model-level loaders
  ## that use this proc keep the collection alive beside the weights.
  ## Raises ValueError naming the tensor when the collection holds no entry.
  view.files[view.fileOf(tensorName)].getTensorView(tensorName)
