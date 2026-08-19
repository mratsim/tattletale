# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## NimGpuFnBuiltins registration guard.
##
## The list is the name-only registration path for function-style magic builtins
## (nim_to_gpu.nim, registerGenericInstOrExternalProc). The OpenCL work-item spellings
## get_global_id, get_group_id, get_local_id, get_local_size and get_num_groups
## are template aliases in builtins_catalog.nim that expand to canonical names
## during sem. get_global_size is OpenCL-native and excluded from the vocabulary,
## so none of them ever reaches this list. This test pins the list to its two legitimate entries
## and forbids re-adding any `get_*` name.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_builtin_registration_guard.nim

import std/strutils
import workspace/crucible/src/codegen/builtins/nim_builtins

static:
  doAssert NimGpuFnBuiltins.len == 2,
    "NimGpuFnBuiltins must hold exactly toOpenArray and len, got: " & $NimGpuFnBuiltins
  doAssert "toOpenArray" in NimGpuFnBuiltins
  doAssert "len" in NimGpuFnBuiltins
  for name in NimGpuFnBuiltins:
    doAssert not name.startsWith("get_"),
      "NimGpuFnBuiltins must not register OpenCL work-item spellings, got: " & name
  echo "  OK — NimGpuFnBuiltins holds only toOpenArray and len (no get_* entry)"

when isMainModule:
  echo ""
  echo "  All builtin registration guard tests passed."
