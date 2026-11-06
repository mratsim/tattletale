# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os
import workspace/crucible/src/abis/objc_abi as objc

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

const
  ## Shared + untracked storage, Apple's MLX allocator mode: the host reads contents() directly.
  MTLResourceStorageModeShared = 0
  MTLResourceHazardTrackingModeUntracked = 1 shl 8
  SharedUntrackedOptions* = objc.NSUInteger(MTLResourceStorageModeShared or
                                           MTLResourceHazardTrackingModeUntracked)

  ## `MTLLanguageVersion` enum values (MTLLibrary.h): 3_1 = macOS 14,
  ## 3_2 = macOS 15, 4_0 = macOS 26.
  MTLLanguageVersion3_1 = objc.NSUInteger(0x30001)
  MTLLanguageVersion3_2 = objc.NSUInteger(0x30002)
  MTLLanguageVersion4_0 = objc.NSUInteger(0x40000)

# ═══════════════════════════════════════════════════════════════════════
# Error records
# ═══════════════════════════════════════════════════════════════════════

type
  MetalContext* = ref MetalContextObj
  MetalContextObj = object
    ## Live Metal device and command queue, +1 objects owned by the caller.
    device*: objc.ID
    queue*: objc.ID

proc `=copy`(dst: var MetalContextObj, src: MetalContextObj) {.error:
    "MetalContext owns its device and queue handles, copy the MetalEngine ref not the ctx".}

when defined(macosx):
  # objc_abi's withMemPool/release are macOS-only: Linux builds of the engines layer
  # fail to compile this destructor when the `when defined(macosx)` check omits it.
  proc `=destroy`(ctx: var MetalContextObj) =
      objc.withMemPool:
        objc.release(ctx.queue)
        objc.release(ctx.device)

# ═══════════════════════════════════════════════════════════════════════
# macOS-only execution primitives
# ═══════════════════════════════════════════════════════════════════════

when defined(macosx):
  # ═════════════════════════════════════════════════
  # MSL version ladder (uname)
  # ═════════════════════════════════════════════════

  type
    UtSName {.importc: "struct utsname", header: "<sys/utsname.h>".} = object
      ## libc `struct utsname`: every field is 256 chars on macOS.
      sysname: array[256, char]
      nodename: array[256, char]
      release: array[256, char]
      version: array[256, char]
      machine: array[256, char]

  proc uname(uts: ptr UtSName): cint {.importc, header: "<sys/utsname.h>".}

  proc getMslLanguageVersion(): objc.NSUInteger =
    ## MSL language version for this OS, from the `uname()` Darwin release:
    ## < 24 → 3.1, 24 → 3.2, ≥ 25 → 4.0 (NSProcessInfo is sret, UB via msgSend).
    var uts: UtSName
    if uname(addr uts) != 0:
      raiseAssert("uname failed, cannot determine the MSL language version")
    var major = 0
    for c in uts.release:
      if c in {'0' .. '9'}:
        major = major * 10 + (ord(c) - ord('0'))
      else:
        break
    if major == 0:
      raiseAssert("uname release '" & $cast[cstring](addr uts.release[0]) &
               "' has no leading digit, cannot determine the MSL language version")
    if major < 24:
      result = MTLLanguageVersion3_1
    elif major < 25:
      result = MTLLanguageVersion3_2
    else:
      result = MTLLanguageVersion4_0

  # ═════════════════════════════════════════════════
  # Compilation
  # ═════════════════════════════════════════════════

  proc compileOptions*(): objc.ID =
    let cls = objc.getClass("MTLCompileOptions")
    if objc.isNil(cls):
      raiseAssert("getClass(MTLCompileOptions) is nil, Metal.framework not loaded")
    result = objc.msgSend(objc.ID(cls), objc.`$$`("alloc"))
    # ObjC `init` may return a different object than `alloc`.
    result = objc.msgSend(result, objc.`$$`("init"))
    discard objc.msgSend(result, objc.`$$`("setFastMathEnabled:"), objc.BOOL(0))
    discard objc.msgSend(result, objc.`$$`("setLanguageVersion:"), getMslLanguageVersion())

  proc compileLibrary*(device: objc.ID, source: string, opts: objc.ID): objc.ID =
    var compileError: objc.ID = objc.ID(nil)
    result = objc.msgSend(device, objc.`$$`("newLibraryWithSource:options:error:"),
                          objc.nsStringFromNimString(source), opts, addr compileError)
    if objc.isNil(result):
      var detail = "no NSError object provided"
      if not objc.isNil(compileError):
        detail = objc.nsStringToNimString(objc.msgSend(compileError, objc.`$$`("localizedDescription")))
      raiseAssert("Metal library compilation failed: " & detail)

  proc compilePipelineState*(device, library: objc.ID, kernel: string): objc.ID =
    ## Creates the +1 compute pipeline state for `kernel`.
    let fn = objc.msgSend(library, objc.`$$`("newFunctionWithName:"), objc.nsStringFromNimString(kernel))
    if objc.isNil(fn):
      raiseAssert("newFunctionWithName(" & kernel & ") returned nil")
    defer:
      objc.release(fn)
    var psoError: objc.ID = objc.ID(nil)
    result = objc.msgSend(device, objc.`$$`("newComputePipelineStateWithFunction:error:"),
                          fn, addr psoError)
    if objc.isNil(result):
      var detail = "no NSError object provided"
      if not objc.isNil(psoError):
        detail = objc.nsStringToNimString(objc.msgSend(psoError, objc.`$$`("localizedDescription")))
      raiseAssert("compute pipeline state creation failed for " & kernel & ": " & detail)

