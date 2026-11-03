# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile IR parser: loud subset refusals
##
## Test that constructs outside the supported subset refuse through
## SubsetError, with the construct and the kernel named in the message.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_ir_refusals.nim

import std/strutils

import workspace/tattletile/tattletile

proc refuses(text, construct, kernel: string) =
  ## Parses `text` and requires a SubsetError whose message names
  ## `construct` and `kernel`.
  try:
    discard parseTileIr(text)
    doAssert false, "expected a refusal naming '" & construct & "'"
  except SubsetError as e:
    doAssert construct in e.msg,
      "refusal does not name '" & construct & "': " & e.msg
    doAssert kernel in e.msg,
      "refusal does not name the kernel '" & kernel & "': " & e.msg

let head = "cuda_tile.module @k { entry @mykernel(%p: tile<ptr<f32>>) {"
let tail = " } }"

# unsupported op constructs, named verbatim
refuses(head & " %g = make_gather_scatter_view %p : x" & tail, "make_gather_scatter_view", "mykernel")
refuses(head & " %a = mmai %p, %p, %p : tile<4xi8>" & tail, "mmai", "mykernel")
refuses(head & " %x = shuffle %p : tile<4xf32>" & tail, "shuffle", "mykernel")
refuses(head & " %v = make_strided_view %p : x" & tail, "make_strided_view", "mykernel")

# unsupported element and constant types, named verbatim
refuses(head & " %a = addf %p, %p : tile<4xf16>" & tail, "f16", "mykernel")
refuses(head & " %a = addf %p, %p : tile<4xbf16>" & tail, "bf16", "mykernel")
refuses(head & " %a = constant <f32: 1.0> : tile<f8>" & tail, "f8", "mykernel")
refuses(head & " %p2 = make_partition_view %p : partition_view<tile=(?), tensor_view<?xf32, strides=[?]>>" & tail,
        "partition_view with dynamic tile dims", "mykernel")
refuses(head & " %a = constant <f16: 1.0> : tile<4xf16>" & tail, "f16", "mykernel")

# malformed op shapes for the subset
refuses(head & " %x, %y = get_tile_block_id : tile<i32>" & tail, "get_tile_block_id", "mykernel")
refuses(head & " %a = constant <i32: [1, 2]> : tile<4xi32>" & tail, "constant", "mykernel")
refuses(head & " %a = constant <i32: [1, 2, 3, 4]> : tile<?xi32>" & tail, "constant", "mykernel")
refuses(head & " %a = constant <i32: ?> : tile<i32>" & tail, "constant value token", "mykernel")

# compiler-emitted constructs that stay outside the subset
refuses(head & " %a = constant <f32: 1.0> : tile<8xf32>" & " return %a" & tail,
        "return with operands", "mykernel")
refuses(head & " %v = make_partition_view %p : partition_view<tile=(8), padding_value = nan, tensor_view<?xf32, strides=[?]>>" & tail,
        "padding_value", "mykernel")
refuses("cuda_tile.module @k { entry @mykernel(%p: tile<ptr<f32>>) other = 3 { } }",
        "entry attribute", "mykernel")
refuses("cuda_tile.module @k { entry @mykernel(%p: tile<ptr<f32>>) di_name = 3 { } }",
        "entry attribute value", "mykernel")
refuses(head & " continue" & tail, "continue", "mykernel")
refuses(head & " %z = constant <i32: 0> : tile<i32>" &
        " %r = for %i in (%z to %z, step %z) : tile<i32> { yield }" &
        " yield" & tail, "yield", "mykernel")
refuses(head & " %z = constant <i32: 0> : tile<i32>" &
        " %r = for %i in (%z to %z, step %z) : tile<i32>" &
        " { %c = constant <i32: 1> : tile<i32> }" & " yield" & tail,
        "for region without continue", "mykernel")

# integer overflow refuses through the subset refusal, not a raw overflow defect
refuses(head & " %a = constant <f32: 1.0> : tile<99999999999999999999xf32>" & tail, "integer token", "mykernel")

# dynamic shape entries require the `tile<i32> -> ` operand-type prefix
refuses(head & " %tv = make_tensor_view %p, shape = [%n], strides = [1] : tensor_view<?xf32, strides=[?]>" & tail,
        "make_tensor_view", "mykernel")

# constant payload element type must match the tile element type
refuses(head & " %a = constant <f32: 1.0> : tile<4xi32>" & tail, "constant", "mykernel")
refuses(head & " %a = constant <i32: 1> : tile<4xf32>" & tail, "constant", "mykernel")

# module-level refusals carry no kernel, the module placeholder stands in for it
refuses("garbage", "garbage", "<module>")
refuses(head & " %a = addf %p, %p : tile<4xf32>" & tail & " trailing", "trailing text", "<module>")
refuses(head & " %a = constant <f32: 3.0 : tile<f32>" & tail, "expected '>'", "mykernel")
refuses(head & " %a = ", "unexpected end of Tile IR", "mykernel")

# tokenizer refusal for an unterminated string has no kernel context
try:
  discard parseTileIr("cuda_tile.module @k { entry @mykernel(%p: tile<ptr<f32>>) di_name = \"oops { } }")
  doAssert false, "expected a tokenize refusal"
except SubsetError as e:
  doAssert "cannot tokenize" in e.msg, "unexpected refusal: " & e.msg

echo "test_tile_ir_refusals: PASS"
