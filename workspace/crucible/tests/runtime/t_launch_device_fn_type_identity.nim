# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## launchDeviceFn bridge names hash each type's definition site.
##
## The bridge proc name hashes a launch's parameter types, and the batch dedup reuses a wrapper by backend and kernel name.
## Two modules that define types under one identifier must hash differently, or the second launch reuses the first wrapper.
##
## The collision has no runtime observation through the launch surface, so the check runs on the hash input itself:
## - device fns resolve by arity
## - the parameter type comes from the device fn declaration
## - a type symbol is not a value the launch accepts
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn_type_identity.nim

import std/[macros, hashes]
import workspace/crucible/src/runtime/launch/launch_device_fn {.all.}
import ./type_identity_config_a as cfgA
import ./type_identity_config_b as cfgB

macro paramTypeSpelling(v: typed): string =
  ## Name of a launch parameter's type as the bridge-name hash sees it
  result = newLit(v.getTypeInst().strVal)

macro paramTypeHash(v: typed): Hash =
  ## Bridge-name hash input for a launch parameter, as buildGlobalFn sees it
  result = newLit(hashTypeIdentity(v.getTypeInst()))

proc runTest() =
  var a: cfgA.Config
  var b: cfgB.Config

  doAssert paramTypeSpelling(a) == "Config"
  doAssert paramTypeSpelling(b) == "Config"

  let hashA = paramTypeHash(a)
  doAssert paramTypeHash(b) != hashA,
    "two types defined under one identifier in different modules must hash differently"

  var f: float32
  # a builtin has no definition and hashes by the identifier alone
  discard paramTypeHash(f)

  doAssert paramTypeHash(a) == hashA,
    "one parameter must hash stably so legitimate dedup still reuses the wrapper"

  echo "[OK] launchDeviceFn: the bridge-name hash distinguishes types that share an identifier across modules"

when isMainModule:
  runTest()
