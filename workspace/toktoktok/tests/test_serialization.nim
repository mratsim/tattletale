import std/unittest
import std/os
import std/sequtils
import std/tables

import ../src/serialization

const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"

proc runSerializationTests() =
  suite "Serialization Tests":


    test "convertHfToTiktoken handles ASCII tokens":
      let hfJson = """{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        "model": {
          "type": "BPE",
          "vocab": {
            "a": 0,
            "b": 1,
            "c": 2,
            "hello": 3
          }
        }
      }"""
      let hf = deserializeHfTokenizer(hfJson)
      let format = convertHfToTiktoken(hf)

      check format.pattern.regexp.len > 0  # Should have a pattern
      check format.specialTokens.len == 0

      check format.mergeableRanks[@[byte(97)]] == 0
      check format.mergeableRanks[@[byte(98)]] == 1
      check format.mergeableRanks[@[byte(99)]] == 2
      check format.mergeableRanks[@[byte(104), byte(101), byte(108), byte(108), byte(111)]] == 3

    test "convertHfToTiktoken handles special tokens":
      let hfJson = """{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
          {"id": 50256, "special": true, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        "model": {
          "type": "BPE",
          "vocab": {}
        }
      }"""
      let hf = deserializeHfTokenizer(hfJson)
      let format = convertHfToTiktoken(hf)

      check toSeq(format.specialTokens.pairs).len == 1
      check format.specialTokens["<|endoftext|>"] == 50256

    test "convertHfToTiktoken handles empty vocab (adds byte tokens)":
      let hfJson = """{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        "model": {
          "type": "BPE",
          "vocab": {}
        }
      }"""
      let hf = deserializeHfTokenizer(hfJson)
      let format = convertHfToTiktoken(hf)
      check format.mergeableRanks.len == 256  # All byte tokens are added

    test "convertHfToTiktoken handles space token":
      let hfJson = """{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        "model": {
          "type": "BPE",
          "vocab": {
            " ": 10,
            "hello": 20
          }
        }
      }"""
      let hf = deserializeHfTokenizer(hfJson)
      let format = convertHfToTiktoken(hf)

      # Byte 32 (space) should be added with high rank since " " key doesn't convert
      check format.mergeableRanks.hasKey(@[byte(32)])
      check format.mergeableRanks[@[byte(32)]] > 1000000  # High rank for byte tokens
      # "hello" should have rank 20
      check format.mergeableRanks.hasKey(@[byte(104), byte(101), byte(108), byte(108), byte(111)])
      check format.mergeableRanks[@[byte(104), byte(101), byte(108), byte(108), byte(111)]] == 20

    test "convertHfToTiktoken handles special tokens":
      let hfJson = """{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
          {"id": 50256, "special": true, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        "model": {
          "type": "BPE",
          "vocab": {}
        }
      }"""
      let hf = deserializeHfTokenizer(hfJson)
      let format = convertHfToTiktoken(hf)

      check format.specialTokens.len == 1
      check format.specialTokens["<|endoftext|>"] == 50256

    test "convertHfToTiktoken handles empty vocab (adds byte tokens)":
      let hfJson = """{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        "model": {
          "type": "BPE",
          "vocab": {}
        }
      }"""
      let hf = deserializeHfTokenizer(hfJson)
      let format = convertHfToTiktoken(hf)
      check format.mergeableRanks.len == 256  # All byte tokens are added

when isMainModule:
  runSerializationTests()
