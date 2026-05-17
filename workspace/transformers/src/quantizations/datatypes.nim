# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Quantization format types — pure enum, no imports beyond std.
##
## Shared by layers (for object variants) and quantization codecs.
## Deliberately kept separate from all_interfaces to avoid circular deps.

type QuantFormatKind* = enum
  qBF16
    ## Standard BF16 weights — no quantization.
  qExl3
    ## EXL3 quantization: trellis decoded at load time to FP16 (not BF16),
    ## per-token Hadamard transform on input and output.
