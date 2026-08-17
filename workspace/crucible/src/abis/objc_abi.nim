# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Minimal Objective-C runtime bridge for the Metal backend.
##
## Sends ObjC methods through libobjc's `objc_msgSend`, with no Objective-C
## source, no C++ wrapper, no link flags. Every symbol below resolves at
## runtime via `dynlib:`; Nim loads the frameworks at module init, so there
## is no manual init step.
##
## Tested ABI (macOS, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10.
##
## Scope: the msgSend core only. Class reflection (ivars, methods, protocols,
## properties, IMP, exceptions, class-pair allocation) is intentionally
## absent. `MTL*` are protocols, not classes: `objc_getClass("MTLDevice")`
## returns nil; obtain Metal objects from `MTLCreateSystemDefaultDevice` and
## msgSend.

const ObjcLibName = "libobjc.A.dylib"

# ═══════════════════════════════════════════════════════════════════════
# Scalar and opaque types
# ═══════════════════════════════════════════════════════════════════════

when defined(cpu64):
  type
    CGFloat* = cdouble
    NSInteger* = clong
    NSUInteger* = culong
else:
  type
    CGFloat* = cfloat
    NSInteger* = cint
    NSUInteger* = cuint

type
  ID* = distinct pointer
  Class* = distinct pointer
  SEL* = distinct pointer
  BOOL* = cchar

  MTLSize* = object
    ## Metal threadgroup/grid size: three NSUIntegers, 24 bytes on arm64.
    ## Passed by value to `dispatchThreadgroups:threadsPerThreadgroup:`. The
    ## arm64 aggregate ABI routes the struct by reference past `objc_msgSend`.
    width*: NSUInteger
    height*: NSUInteger
    depth*: NSUInteger

proc isNil*(a: ID): bool =
  ## Returns true when the object pointer is null.
  pointer(a) == nil

proc isNil*(a: Class): bool =
  ## Returns true when the class pointer is null.
  pointer(a) == nil

# ═══════════════════════════════════════════════════════════════════════
# msgSend
# ═══════════════════════════════════════════════════════════════════════

{.pragma: objcImport, cdecl, importc, dynlib: ObjcLibName.}

## libobjc `objc_msgSend` symbol, plain varargs import. Never called with
## extra arguments from Nim. Apple arm64 passes C variadic arguments on the
## stack, where `objc_msgSend` does not look for them. The `msgSend` family
## below is the only call path; it casts this symbol to each method's exact
## fixed C signature before invoking.
proc objc_msgSend(self: ID; op: SEL): ID {.objcimport, varargs.}

# Fixed C signatures for the selectors the Metal backend uses. The cast makes
# the C compiler pass the arguments in the registers the method IMP expects
# (x2+), which a variadic call would not. Returns the method result as `ID`;
# selectors returning void or a pointer are fine (discard or cast the result).
# Selectors returning a struct larger than 16 bytes use the x8 sret register
# and must never be sent through this bridge (UB).
type
  MsgSend0 = proc (self: ID; op: SEL): ID {.cdecl.}
  MsgSendCstr = proc (self: ID; op: SEL; a: cstring): ID {.cdecl.}
  MsgSendID = proc (self: ID; op: SEL; a: ID): ID {.cdecl.}
  MsgSendIDPtr = proc (self: ID; op: SEL; a: ID; b: ptr ID): ID {.cdecl.}
  MsgSendIDIDPtr = proc (self: ID; op: SEL; a: ID; b: ID; c: ptr ID): ID {.cdecl.}
  MsgSendUInt2 = proc (self: ID; op: SEL; a: NSUInteger; b: NSUInteger): ID {.cdecl.}
  MsgSendIDUInt2 = proc (self: ID; op: SEL; a: ID; b: NSUInteger; c: NSUInteger): ID {.cdecl.}
  MsgSendSize2 = proc (self: ID; op: SEL; a: MTLSize; b: MTLSize): ID {.cdecl.}

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

## Registers `str` as a selector (runtime-interned, returns the existing
## selector when already registered).
proc sel_registerName(str: cstring): SEL {.objcimport.}

proc `$$`*(str: string): SEL =
  ## Returns the registered selector for `str`, e.g. `$$"alloc"`.
  sel_registerName(str.cstring)

## Returns the class registered under `name`, or nil when not registered.
## Only real classes resolve: `MTL*` are protocols and come back nil.
proc objc_getClass*(name: cstring): Class {.objcimport.}

## Same lookup as `objc_getClass` (identical implementation in libobjc);
## returns nil when the class is not registered.
proc objc_lookUpClass*(name: cstring): Class {.objcimport.}

# ═══════════════════════════════════════════════════════════════════════
# NSString bridge
# ═══════════════════════════════════════════════════════════════════════

proc nsStringToNimString*(ns: ID): string =
  ## Converts an NSString (e.g. NSError `localizedDescription`) to a Nim
  ## string via `UTF8String`. A nil `ns` yields an empty string.
  let utf8 = msgSend(ns, $$"UTF8String")
  result = $cast[cstring](utf8)

# ═══════════════════════════════════════════════════════════════════════
# Framework auto-load
# ═══════════════════════════════════════════════════════════════════════

## Returns the current user's name (an NSString), or nil.
##
## This import exists to load Foundation.framework at module init: `dynlib:`
## symbols resolve eagerly, and NS* classes (NSAutoreleasePool, NSString) are
## nil to `objc_getClass` until the framework is loaded. `NSUserName` is a
## zero-argument C function, so the load trigger below needs no NSString
## (which cannot exist before the load).
proc NSUserName(): ID {.importc, cdecl, dynlib: "/System/Library/Frameworks/Foundation.framework/Foundation".}

## True once Foundation.framework is loaded (always true in practice: a
## failed dynlib load quits at module init). The call itself is what keeps
## the `NSUserName` import alive: Nim dead-code-eliminates unreferenced
## dynlib imports, so without the call the framework would never load.
let foundationLoaded* = not NSUserName().isNil

## Returns the default Metal device, or nil when no Metal device exists.
##
## Imported directly (a plain C function, not a method). The `dynlib:` import
## doubles as the Metal.framework auto-load.
proc MTLCreateSystemDefaultDevice*(): ID {.importc, cdecl, dynlib: "/System/Library/Frameworks/Metal.framework/Metal".}

## Sets the class version number (libobjc `class_setVersion`).
##
## `{.importc: "class_$1".}` substitutes the Nim proc name into the C symbol,
## so the Nim name stays unprefixed: the generated symbol is `class_setVersion`,
## never `class_class_setVersion`.
proc setVersion*(cls: Class; version: cint) {.importc: "class_$1", cdecl, dynlib: ObjcLibName.}
