# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Minimal Objective-C runtime bridge for the Metal backend.

# ═════════════════════════════════════════════════
# Scalar and opaque types (platform-neutral)
# ═════════════════════════════════════════════════

when defined(cpu64):
  type
    NSUInteger* = culong
else:
  type
    NSUInteger* = cuint

type
  # Objective-C object pointer (`id`)
  ID* = distinct pointer
  # Objective-C class object pointer
  Class* = distinct pointer
  # Objective-C selector, a method name interned by the runtime
  SEL* = distinct pointer
  # Objective-C boolean, a C `signed char` (0/1)
  BOOL* = cchar

  MTLSize* = object
    ## Metal threadgroup/grid size: three NSUIntegers, 24 bytes on arm64.
    ## Passed by value to `dispatchThreadgroups:threadsPerThreadgroup:`.
    ## The arm64 aggregate ABI routes the struct by reference past `objc_msgSend`.
    width*: NSUInteger
    height*: NSUInteger
    depth*: NSUInteger

proc isNil*(a: ID): bool {.inline.} =
  ## Returns true when the object pointer is null.
  pointer(a) == nil

proc isNil*(a: Class): bool {.inline.} =
  ## Returns true when the class pointer is null.
  pointer(a) == nil

# ═════════════════════════════════════════════════
# macOS-only surface
# ═════════════════════════════════════════════════

when defined(macosx):
  const ObjcLibName = "libobjc.A.dylib"

  # ═════════════════════════════════════════════════
  # msgSend
  # ═════════════════════════════════════════════════

  {.pragma: objcImport, cdecl, importc, dynlib: ObjcLibName.}

  proc objc_msgSend(self: ID, op: SEL): ID {.objcimport, varargs.}

  type
    MsgSend0 = proc (self: ID, op: SEL): ID {.cdecl.}
    MsgSendCstr = proc (self: ID, op: SEL, a: cstring): ID {.cdecl.}
    MsgSendID = proc (self: ID, op: SEL, a: ID): ID {.cdecl.}
    MsgSendIDPtr = proc (self: ID, op: SEL, a: ID, b: ptr ID): ID {.cdecl.}
    MsgSendIDIDPtr = proc (self: ID, op: SEL, a: ID, b: ID, c: ptr ID): ID {.cdecl.}
    MsgSendIDUIntPtr2 = proc (self: ID, op: SEL, a: ID, b: NSUInteger, c: ptr ID, d: ptr ID): ID {.cdecl.}
    MsgSendUInt2 = proc (self: ID, op: SEL, a: NSUInteger, b: NSUInteger): ID {.cdecl.}
    MsgSendIDUInt2 = proc (self: ID, op: SEL, a: ID, b: NSUInteger, c: NSUInteger): ID {.cdecl.}
    MsgSendPtrUInt2 = proc (self: ID, op: SEL, a: pointer, b: NSUInteger, c: NSUInteger): ID {.cdecl.}
    MsgSendPtrUInt2ID = proc (self: ID, op: SEL, a: pointer, b: NSUInteger, c: NSUInteger, d: ID): ID {.cdecl.}
    MsgSendSize2 = proc (self: ID, op: SEL, a: MTLSize, b: MTLSize): ID {.cdecl.}
    MsgSendBOOL = proc (self: ID, op: SEL, a: BOOL): ID {.cdecl.}
    MsgSendUInt1 = proc (self: ID, op: SEL, a: NSUInteger): ID {.cdecl.}
    MsgSendUInt0 = proc (self: ID, op: SEL): NSUInteger {.cdecl.}

  template msgSend*(self: ID, op: SEL): ID =
    cast[MsgSend0](objc_msgSend)(self, op)

  template msgSend*(self: ID, op: SEL, a: cstring): ID =
    cast[MsgSendCstr](objc_msgSend)(self, op, a)

  template msgSend*(self: ID, op: SEL, a: ID): ID =
    cast[MsgSendID](objc_msgSend)(self, op, a)

  template msgSend*(self: ID, op: SEL, a: ID, b: ptr ID): ID =
    cast[MsgSendIDPtr](objc_msgSend)(self, op, a, b)

  template msgSend*(self: ID, op: SEL, a: ID, b: ID, c: ptr ID): ID =
    cast[MsgSendIDIDPtr](objc_msgSend)(self, op, a, b, c)

  template msgSend*(self: ID, op: SEL, a: ID, b: NSUInteger, c: ptr ID, d: ptr ID): ID =
    cast[MsgSendIDUIntPtr2](objc_msgSend)(self, op, a, b, c, d)

  template msgSend*(self: ID, op: SEL, a: NSUInteger, b: NSUInteger): ID =
    cast[MsgSendUInt2](objc_msgSend)(self, op, a, b)

  template msgSend*(self: ID, op: SEL, a: ID, b: NSUInteger, c: NSUInteger): ID =
    cast[MsgSendIDUInt2](objc_msgSend)(self, op, a, b, c)

  template msgSend*(self: ID, op: SEL, a: pointer, b: NSUInteger, c: NSUInteger): ID =
    cast[MsgSendPtrUInt2](objc_msgSend)(self, op, a, b, c)

  template msgSend*(self: ID, op: SEL, a: pointer, b: NSUInteger, c: NSUInteger, d: ID): ID =
    cast[MsgSendPtrUInt2ID](objc_msgSend)(self, op, a, b, c, d)

  template msgSend*(self: ID, op: SEL, a: MTLSize, b: MTLSize): ID =
    cast[MsgSendSize2](objc_msgSend)(self, op, a, b)

  template msgSend*(self: ID, op: SEL, a: BOOL): ID =
    cast[MsgSendBOOL](objc_msgSend)(self, op, a)

  template msgSend*(self: ID, op: SEL, a: NSUInteger): ID =
    cast[MsgSendUInt1](objc_msgSend)(self, op, a)

  template msgSendUInt*(self: ID, op: SEL): NSUInteger =
    cast[MsgSendUInt0](objc_msgSend)(self, op)

  proc selRegisterName(str: cstring): SEL {.objcimport, importc: "sel_registerName".}

  proc `$$`*(str: string): SEL {.inline.} =
    ## Returns the registered selector for `str`, `$$"alloc"` for example.
    selRegisterName(str.cstring)

  proc getClass*(name: cstring): Class {.objcimport, importc: "objc_$1".}

  proc lookUpClass*(name: cstring): Class {.objcimport, importc: "objc_$1".}

  # ═════════════════════════════════════════════════
  # NSString bridge
  # ═════════════════════════════════════════════════

  proc nsStringToNimString*(ns: ID): string =
    ## Converts an NSString (e.g. NSError `localizedDescription`) to a Nim string
    ## via `UTF8String`. A nil `ns` yields an empty string.
    let utf8 = msgSend(ns, $$"UTF8String")
    result = $cast[cstring](utf8)

  proc nsStringFromNimString*(s: string): ID =
    ## Builds an autoreleased NSString from `s` via `stringWithUTF8String:`.
    let cls = getClass("NSString")
    doAssert not cls.isNil, "NSString is nil, Foundation not loaded"
    doAssert '\0' notin s, "string contains NUL, cannot bridge to NSString"
    msgSend(ID(cls), $$"stringWithUTF8String:", s.cstring)

  # ═════════════════════════════════════════════════
  # Memory management
  # ═════════════════════════════════════════════════

  proc newPool*(): ID {.inline.} =
    ## Allocates and inits an `NSAutoreleasePool`, engine calls wrap their body in one.
    ## Without a pool, autoreleased Metal objects trip OBJC_DEBUG_MISSING_POOLS=YES.
    result = msgSend(ID(getClass("NSAutoreleasePool")), $$"alloc")
    # `init` may return a different object than `alloc`, keep the init result.
    result = msgSend(result, $$"init")

  proc drainPool*(pool: ID) {.inline.} =
    ## Drains the pool, releases everything autoreleased since `newPool`.
    discard msgSend(pool, $$"drain")

  template withMemPool*(body: untyped): untyped =
    ## Runs `body` inside a fresh autorelease pool, drained on scope exit.
    ## The drain releases every autoreleased object the body created.
    let pool = newPool()
    defer: pool.drainPool()
    body

  proc release*(obj: ID) {.inline.} =
    ## Releases a +1 Objective-C object, balanced against a `new*`/Create acquisition, nil-safe.
    if not isNil(obj):
      discard msgSend(obj, $$"release")

  # ═════════════════════════════════════════════════
  # Framework auto-load
  # ═════════════════════════════════════════════════

  import workspace/cpuplatforms/loadtime_functions

  proc dlopen(path: cstring, mode: cint): pointer {.importc, header: "<dlfcn.h>".}
  const RtlNow = 2  # RTLD_NOW on macOS (and Linux)

  proc loadFoundation() {.loadTime.} =
    ## Loads Foundation.framework at program load time, before main and module init.
    ## From here on, NS* classes are visible to `objc.getClass` and `objc.lookUpClass`.
    discard dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", RtlNow)

  proc MTLCreateSystemDefaultDevice*(): ID {.importc: "MTLCreateSystemDefaultDevice", cdecl, dynlib: "/System/Library/Frameworks/Metal.framework/Metal".}

