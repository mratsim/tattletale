# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Device selection for the transformer suites.
##
## Suites resolve the compute device through one code path, `testDevice()`,
## instead of hardcoding a device per suite. The compile-time string define
## `TTT_TEST_ON` (auto | metal | cpu | cuda) selects it.
##
## An explicit value wins over the platform default:
##
## - auto, Metal on macOS, CUDA when a Linux host provides one
##   (`Torch.cuda_is_available()`), CPU as the last resort
## - metal, cpu or cuda, the named device on any platform
##
## The per-suite test tasks translate an environment `TTT_TEST_ON` value into
## this define through config.nims, so `TTT_TEST_ON=cpu nim test_<suite>` flips
## the device without a hand-written build command.

import
  workspace/libtorch as F

const
  TTT_TEST_ON* {.strdefine.} = "auto"
    ## Compile-time device selection, one of auto, metal, cpu or cuda.

proc parseTestDevice*(name: string): F.DeviceKind =
  ## Returns the device named by one TTT_TEST_ON value. Unknown names raise.
  case name
  of "metal": F.kMPS
  of "cpu": F.kCPU
  of "cuda": F.kCUDA
  else:
    raise newException(ValueError,
      "TTT_TEST_ON must name auto, metal, cpu or cuda, got: " & name)

proc testDevice*(): F.DeviceKind =
  ## Returns the suite compute device.
  ##
  ## - an explicit TTT_TEST_ON value wins
  ## - auto, Metal on macOS, CUDA when a Linux host provides one, CPU otherwise
  ##
  ## On a non-macOS host the auto default asks `Torch.cuda_is_available()`
  ## at run time inside the compiled suite. Apple Silicon always takes
  ## the Metal branch, so the fallback branch is verified by the define logic alone.
  when TTT_TEST_ON == "auto":
    when defined(macosx):
      F.kMPS
    else:
      if F.Torch.cuda_is_available(): F.kCUDA else: F.kCPU
  else:
    parseTestDevice(TTT_TEST_ON)

proc deviceName*(device: F.DeviceKind): string =
  ## Stable printed name of a device kind, used by suite reports.
  case device
  of F.kCPU: "cpu"
  of F.kCUDA: "cuda"
  of F.kMPS: "metal"
  else: $device
