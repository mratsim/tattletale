## Manual test: `var result` at outer scope is rejected
##
## Expected compile-time error:
##   shadowPlain has a custom `result` variable which shadows
##   the implicit `result` (not allowed in GPU code)
##
## Run with: nim cpp -d:cuda workspace/crucible/tests/codegen/nvrtc/manual_nvrtc_reject_result_shadow.nim
import workspace/crucible/src/codegen/nvrtc

const k = cuda:
  proc shadowPlain(x: uint32): uint32 {.device.} =
    var result = x * 2
    return result
