# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Minimal Objective-C runtime bridge for the Metal backend.
##
## Sends ObjC methods through libobjc's `objc_msgSend`, with no Objective-C source,
## no C++ wrapper, and no link flags.
## Symbol resolution is split by mechanism. `dynlib:` imports load libobjc
## and Metal.framework at Nim module init. A load-time constructor
## opens Foundation.framework before `main`, so NS* classes are visible
## to `getClass` and `lookUpClass` immediately.
##
## Tested ABI (macOS, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10.
##
## Scope: the msgSend core only. Class reflection is intentionally absent
## (ivars, methods, protocols, properties, IMP, exceptions, class-pair allocation).
## `MTL*` are protocols, not classes, so `getClass("MTLDevice")`
## returns nil. Obtain Metal objects from `MTLCreateSystemDefaultDevice`
## and msgSend. Memory management completes the bridge: autorelease pools
## (`newPool`/`drainPool`/`withMemPool`) wrap engine call bodies, and
## `release` balances +1 `new*`/Create acquisitions.
##
## The module is importable on any platform. The scalar and opaque types
## (ID, Class, SEL, BOOL, MTLSize, NSUInteger) and the `isNil` checks are
## platform-neutral, while every macOS-only symbol (msgSend, class lookup,
## the NSString bridge, the framework auto-load) sits behind
## `when defined(macosx)`. Module init therefore performs no dynlib
## loading on Linux or other platforms.
##
## Consumers import this module as `import objc_abi as objc` and call every
## symbol with the `objc.` prefix: `objc.msgSend`, `objc.getClass`,
## `objc.ID`, `objc.MTLSize`, and so on.

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
  ## Objective-C object pointer (`id`).
  ID* = distinct pointer
  ## Objective-C class object pointer.
  Class* = distinct pointer
  ## Objective-C selector: a method name, interned by the runtime.
  SEL* = distinct pointer
  ## Objective-C boolean: a C `signed char` (0/1).
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

  # libobjc `objc_msgSend` symbol: a plain varargs import.
  # Never called with extra arguments from Nim.
  # Apple arm64 passes C variadic arguments on the stack, where the
  # `objc_msgSend` assembly stub does not look for them. The stub reads
  # x0/x1, looks up the IMP, and tail-calls it, with the method arguments
  # in x2+ (GPR) and d0+ (SIMD) per the fixed method signature.
  # Machine-verified on Apple M4 (arm64): a variadic call with one extra
  # cstring argument crashes with SIGSEGV, because the generated call site
  # stores the argument on the stack while the IMP reads x2. Doubles work
  # by accident, because the first SIMD argument lands in d0 under both
  # conventions. The `msgSend` family below is the only call path.
  # It casts this symbol to each method's exact fixed C signature
  # before invoking.
  proc objc_msgSend(self: ID; op: SEL): ID {.objcimport, varargs.}

  # Fixed C signatures for the selectors the Metal backend uses.
  # The cast makes the C compiler pass the arguments in the registers
  # the method IMP expects (x2+), which a variadic call would not.
  # Returns the method result as `ID`.
  # Selectors returning void or a pointer are fine (discard or cast the result).
  # arm64 needs no `objc_msgSend_stret`/`_fpret` variants.
  # The fixed cast handles pointer and id returns.
  # Struct returns larger than 16 bytes use the x8 struct-return register
  # and must never be sent through this bridge (undefined behavior).
  # None of the Metal selectors used here return such structs.
  type
    MsgSend0 = proc (self: ID; op: SEL): ID {.cdecl.}
    MsgSendCstr = proc (self: ID; op: SEL; a: cstring): ID {.cdecl.}
    MsgSendID = proc (self: ID; op: SEL; a: ID): ID {.cdecl.}
    MsgSendIDPtr = proc (self: ID; op: SEL; a: ID; b: ptr ID): ID {.cdecl.}
    MsgSendIDIDPtr = proc (self: ID; op: SEL; a: ID; b: ID; c: ptr ID): ID {.cdecl.}
    MsgSendUInt2 = proc (self: ID; op: SEL; a: NSUInteger; b: NSUInteger): ID {.cdecl.}
    MsgSendIDUInt2 = proc (self: ID; op: SEL; a: ID; b: NSUInteger; c: NSUInteger): ID {.cdecl.}
    MsgSendSize2 = proc (self: ID; op: SEL; a: MTLSize; b: MTLSize): ID {.cdecl.}
    MsgSendBOOL = proc (self: ID; op: SEL; a: BOOL): ID {.cdecl.}
    MsgSendUInt1 = proc (self: ID; op: SEL; a: NSUInteger): ID {.cdecl.}
    MsgSendUInt0 = proc (self: ID; op: SEL): NSUInteger {.cdecl.}

  ## Zero-argument selectors: alloc, init, drain, UTF8String, contents,
  ## newCommandQueue, commandBuffer, computeCommandEncoder, endEncoding, commit,
  ## waitUntilCompleted, localizedDescription.
  template msgSend*(self: ID; op: SEL): ID =
    cast[MsgSend0](objc_msgSend)(self, op)

  ## Selectors with one C-string argument: stringWithUTF8String:.
  template msgSend*(self: ID; op: SEL; a: cstring): ID =
    cast[MsgSendCstr](objc_msgSend)(self, op, a)

  ## Selectors with one object argument: setComputePipelineState:.
  template msgSend*(self: ID; op: SEL; a: ID): ID =
    cast[MsgSendID](objc_msgSend)(self, op, a)

  ## Selectors with one object and one error-out argument:
  ## newComputePipelineStateWithFunction:error:.
  template msgSend*(self: ID; op: SEL; a: ID; b: ptr ID): ID =
    cast[MsgSendIDPtr](objc_msgSend)(self, op, a, b)

  ## Selectors with two objects and one error-out argument:
  ## newLibraryWithSource:options:error:.
  template msgSend*(self: ID; op: SEL; a: ID; b: ID; c: ptr ID): ID =
    cast[MsgSendIDIDPtr](objc_msgSend)(self, op, a, b, c)

  ## Selectors with two NSUInteger arguments: newBufferWithLength:options:.
  template msgSend*(self: ID; op: SEL; a: NSUInteger; b: NSUInteger): ID =
    cast[MsgSendUInt2](objc_msgSend)(self, op, a, b)

  ## Selectors with one object and two NSUInteger arguments:
  ## setBuffer:offset:atIndex:.
  template msgSend*(self: ID; op: SEL; a: ID; b: NSUInteger; c: NSUInteger): ID =
    cast[MsgSendIDUInt2](objc_msgSend)(self, op, a, b, c)

  ## Selectors with two MTLSize arguments:
  ## dispatchThreadgroups:threadsPerThreadgroup:.
  template msgSend*(self: ID; op: SEL; a: MTLSize; b: MTLSize): ID =
    cast[MsgSendSize2](objc_msgSend)(self, op, a, b)

  ## Selectors with one BOOL argument: setFastMathEnabled:.
  template msgSend*(self: ID; op: SEL; a: BOOL): ID =
    cast[MsgSendBOOL](objc_msgSend)(self, op, a)

  ## Selectors with one NSUInteger argument: setLanguageVersion:.
  template msgSend*(self: ID; op: SEL; a: NSUInteger): ID =
    cast[MsgSendUInt1](objc_msgSend)(self, op, a)

  ## Selectors returning an NSUInteger with no extra arguments: status.
  ## Named distinctly from `msgSend(self, op)` because Nim cannot overload
  ## on the return type alone.
  template msgSendUInt*(self: ID; op: SEL): NSUInteger =
    cast[MsgSendUInt0](objc_msgSend)(self, op)

  # Registers `str` as a selector (runtime-interned, returns the existing selector when already registered).
  proc selRegisterName(str: cstring): SEL {.objcimport, importc: "sel_registerName".}

  proc `$$`*(str: string): SEL {.inline.} =
    ## Returns the registered selector for `str`, e.g. `$$"alloc"`.
    selRegisterName(str.cstring)

  ## Returns the class registered under `name`, or nil when not registered.
  ## Only real classes resolve, because `MTL*` are protocols and come back nil.
  ## The C symbol stays `objc_getClass` (the `objc_$1` import pattern).
  proc getClass*(name: cstring): Class {.objcimport, importc: "objc_$1".}

  ## Same lookup as `getClass` (identical implementation in libobjc).
  ## Returns nil when the class is not registered.
  ## The C symbol stays `objc_lookUpClass` (the `objc_$1` import pattern).
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
    doAssert not cls.isNil, "NSString is nil — Foundation not loaded"
    doAssert '\0' notin s, "string contains NUL — cannot bridge to NSString"
    msgSend(ID(cls), $$"stringWithUTF8String:", s.cstring)

  # ═════════════════════════════════════════════════
  # Memory management
  # ═════════════════════════════════════════════════

  proc newPool*(): ID {.inline.} =
    ## Alloc/init an `NSAutoreleasePool`. Engine calls wrap their body in one;
    ## without a pool, autoreleased Metal objects trip OBJC_DEBUG_MISSING_POOLS=YES.
    result = msgSend(ID(getClass("NSAutoreleasePool")), $$"alloc")
    # `init` may return a different object than `alloc` — keep the init result.
    result = msgSend(result, $$"init")

  proc drainPool*(pool: ID) {.inline.} =
    ## Drains the pool: releases everything autoreleased since `newPool`.
    discard msgSend(pool, $$"drain")

  template withMemPool*(body: untyped): untyped =
    ## Runs `body` inside a fresh autorelease pool, drained on scope exit.
    ## The drain releases every autoreleased object the body created.
    let pool = newPool()
    defer: pool.drainPool()
    body

  proc release*(obj: ID) {.inline.} =
    ## Releases a +1 Objective-C object, balanced against a `new*`/Create acquisition. Nil-safe.
    if not isNil(obj):
      discard msgSend(obj, $$"release")

  # ═════════════════════════════════════════════════
  # Framework auto-load
  # ═════════════════════════════════════════════════

  import workspace/cpuplatforms/loadtime_functions

  proc dlopen(path: cstring; mode: cint): pointer {.importc, header: "<dlfcn.h>".}
  const RtlNow = 2  # RTLD_NOW on macOS (and Linux)

  proc loadFoundation() {.loadTime.} =
    ## Loads Foundation.framework at program load time, before main and
    ## before Nim's module-init dynlib loading. From here on, NS* classes
    ## are visible to `objc.getClass` and `objc.lookUpClass`.
    discard dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", RtlNow)

  ## Returns the default Metal device, or nil when no Metal device exists.
  ##
  ## Imported directly (a plain C function, not a method). The `dynlib:` import
  ## doubles as the Metal.framework auto-load.
  proc MTLCreateSystemDefaultDevice*(): ID {.importc: "MTLCreateSystemDefaultDevice", cdecl, dynlib: "/System/Library/Frameworks/Metal.framework/Metal".}

  ## Sets the class version number (libobjc `class_setVersion`).
  ##
  ## `{.importc: "class_$1".}` substitutes the Nim proc name into the C symbol,
  ## so the Nim name stays unprefixed:
  ## the generated symbol is `class_setVersion`, never `class_class_setVersion`.
  proc setVersion*(cls: Class; version: cint) {.importc: "class_$1", cdecl, dynlib: ObjcLibName.}
