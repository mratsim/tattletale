# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Unified error policy for the runtime: stacktrace + stderr + quit(1).

template failLoud*(msg: string) =
  ## Stacktrace + stderr + quit(1) with the caller's location.
  ## A template so instantiationInfo() reports the call site.
  writeStackTrace()
  stderr.write($instantiationInfo() & " exited with error: " & msg & '\n')
  quit 1
