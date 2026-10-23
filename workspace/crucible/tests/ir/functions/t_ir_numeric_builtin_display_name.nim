## numericBuiltinDisplayName
##
## Test that the fnTab display name recognizer strips the whole mangled
## remainder after a `min`/`max`/`abs` stem
##
## - the npHashSuffix tail resolves (`max_RJ1UXDu` -> `max`)
## - Nim's full instantiation mangle resolves (`max___ZE4xeJFcdj6wQZpDx9bBlvQ_SStiky4` -> `max`)
## - a stem-prefixed user function (`max_helper`) never resolves, with any mangled tail
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/functions/t_ir_numeric_builtin_display_name.nim

import workspace/crucible/src/builtins/builtins_functions

static:
  doAssert numericBuiltinDisplayName("max") == "max"
  doAssert numericBuiltinDisplayName("min") == "min"
  doAssert numericBuiltinDisplayName("abs") == "abs"
  doAssert numericBuiltinDisplayName("max_RJ1UXDu") == "max"
  doAssert numericBuiltinDisplayName("min_eegtD5U") == "min"
  doAssert numericBuiltinDisplayName("abs_SStiky4") == "abs"
  # Nim's full instantiation mangle: signatureHash chunks plus the npHashSuffix tail
  doAssert numericBuiltinDisplayName("max___ZE4xeJFcdj6wQZpDx9bBlvQ_SStiky4") == "max"
  doAssert numericBuiltinDisplayName(
    "min___6a1TPDr7NcTeRCpToU9aa0Q___djtm1mkxyUnMHpFQkoGeJA_SStiky4") == "min"
  doAssert numericBuiltinDisplayName("max_RJ1UXDu_ZZZZZZZ") == "max"
  # the stem must be the whole word
  doAssert numericBuiltinDisplayName("maximize_RJ1UXDu") == ""
  doAssert numericBuiltinDisplayName("maxhelper_RJ1UXDu") == ""
  # a user function's word chunks never resolve, with any mangled tail
  doAssert numericBuiltinDisplayName("max_helper") == ""
  doAssert numericBuiltinDisplayName("max_helper_RJ1UXDu") == ""
  doAssert numericBuiltinDisplayName("max_helper_ZE4xeJFcdj6wQZpDx9bBlvQ_SStiky4") == ""
  doAssert numericBuiltinDisplayName("max_") == ""
