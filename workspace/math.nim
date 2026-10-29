# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Math constants and basic-type arithmetic

const Log2e* = 1.44269504089'f32
  ## log₂(e), converts exp to exp2: exp(x) = exp2(x · log₂e).

func ceil_div*(a, b: int): int {.inline.} =
  (a + b - 1) div b

func ceil_div*(a, b: static int): static int {.inline.} =
  (a + b - 1) div b

func round_multiple_up*(a, b: int): int {.inline.} =
  ceil_div(a, b) * b

func round_multiple_up*(a, b: static int): static int {.inline.} =
  ceil_div(a, b) * b

func round_multiple_down*(a, b: int): int {.inline.} =
  (a div b) * b

func round_multiple_down*(a, b: static int): static int {.inline.} =
  (a div b) * b

func sign*(x: int): int {.inline.} =
  if x > 0:
    1
  elif x < 0:
    -1
  else:
    0

func sign*(x: static int): static int {.inline.} =
  if x > 0:
    1
  elif x < 0:
    -1
  else:
    0
