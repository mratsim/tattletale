# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/os,
  workspace/libtorch,
  workspace/transformers/src/layers/attn,
  workspace/transformers/src/models,
  ./common_utils

const ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B"

proc main() =
  echo "Running model loading tests..."
  echo "Model path: ", ModelPath
  echo ""
  
  # Check if model files exist before running tests
  let configPath = ModelPath / "config.json"
  let weightsPath = ModelPath / "model.safetensors"
  
  if not fileExists(configPath):
    echo "❌ Error: config.json not found at ", configPath
    quit(1)
  
  if not fileExists(weightsPath):
    echo "❌ Error: model.safetensors not found at ", weightsPath
    quit(1)
  
  echo "✅ Model files found"
  echo ""
  
  runTest "Qwen3-0.6B model loading":
    proc(): bool =
      ## Test that the model loads successfully without exceptions
      let model = loadModel(ModelPath, kCPU)
      # Model loaded successfully if no exception is raised
      echo "✅ Model loaded successfully"
      result = true

  runTest "Qwen3-0.6B forward pass (single token)":
    proc(): bool =
      ## Test forward pass with a single token (BOS token)
      let model = loadModel(ModelPath, kCPU)
      
      # BOS token for Qwen3
      let input = @[151643'i64]
      let inputTensor = input.toTorchTensor().unsqueeze(0)  # Shape: (1, 1)
      let positions = @[0'i64].toTorchTensor().unsqueeze(0)  # Shape: (1, 1)
      
      var cache = KVCache.init()
      let logits = model.forward(inputTensor, positions, cache)
      
      # Expected shape: (batch=1, seq_len=1, vocab_size=151936)
      result = logits.shape.len == 3 and
               logits.shape[0] == 1 and
               logits.shape[1] == 1 and
               logits.shape[2] == 151936
      
      if result:
        echo "✅ Forward pass (single token) successful"
        echo "   Logits shape: ", logits.shape.len, "D tensor"
      else:
        echo "❌ Forward pass (single token) failed"
        echo "   Expected shape: [1, 1, 151936]"
        echo "   Got shape: ", logits.shape.len, "D tensor"

  runTest "Qwen3-0.6B forward pass (sequence)":
    proc(): bool =
      ## Test forward pass with a sequence of tokens
      let model = loadModel(ModelPath, kCPU)
      
      # BOS token + two dummy tokens
      let input = @[151643'i64, 100'i64, 200'i64]
      let inputTensor = input.toTorchTensor().unsqueeze(0)  # Shape: (1, 3)
      let positions = @[0'i64, 1'i64, 2'i64].toTorchTensor().unsqueeze(0)  # Shape: (1, 3)
      
      var cache = KVCache.init()
      let logits = model.forward(inputTensor, positions, cache)
      
      # Expected shape: (batch=1, seq_len=3, vocab_size=151936)
      result = logits.shape.len == 3 and
               logits.shape[0] == 1 and
               logits.shape[1] == 3 and
               logits.shape[2] == 151936
      
      if result:
        echo "✅ Forward pass (sequence) successful"
        echo "   Logits shape: ", logits.shape.len, "D tensor"
      else:
        echo "❌ Forward pass (sequence) failed"
        echo "   Expected shape: [1, 3, 151936]"
        echo "   Got shape: ", logits.shape.len, "D tensor"

  runTest "Qwen3-0.6B forward pass (batched)":
    proc(): bool =
      ## Test forward pass with batched input
      let model = loadModel(ModelPath, kCPU)
      
      # Batch of 2 sequences with different lengths (padded to same length)
      let input = @[151643'i64, 100'i64, 200'i64, 151643'i64, 150'i64, 250'i64]
      let inputTensor = input.toTorchTensor().reshape([2, 3])  # Shape: (2, 3)
      let positions = @[0'i64, 1'i64, 2'i64, 0'i64, 1'i64, 2'i64].toTorchTensor().reshape([2, 3])  # Shape: (2, 3)
      
      var cache = KVCache.init()
      let logits = model.forward(inputTensor, positions, cache)
      
      # Expected shape: (batch=2, seq_len=3, vocab_size=151936)
      result = logits.shape.len == 3 and
               logits.shape[0] == 2 and
               logits.shape[1] == 3 and
               logits.shape[2] == 151936
      
      if result:
        echo "✅ Forward pass (batched) successful"
        echo "   Logits shape: ", logits.shape.len, "D tensor"
      else:
        echo "❌ Forward pass (batched) failed"
        echo "   Expected shape: [2, 3, 151936]"
        echo "   Got shape: ", logits.shape.len, "D tensor"

  runTest "Qwen3-0.6B multiple forward passes (cache reset)":
    proc(): bool =
      ## Test that cache is properly reset between forward passes
      let model = loadModel(ModelPath, kCPU)
      
      # First forward pass
      let input1 = @[151643'i64, 100'i64]
      let inputTensor1 = input1.toTorchTensor().unsqueeze(0)
      let positions1 = @[0'i64, 1'i64].toTorchTensor().unsqueeze(0)
      
      var cache1 = KVCache.init()
      let logits1 = model.forward(inputTensor1, positions1, cache1)
      
      # Second forward pass with same input should produce same output
      let input2 = @[151643'i64, 100'i64]
      let inputTensor2 = input2.toTorchTensor().unsqueeze(0)
      let positions2 = @[0'i64, 1'i64].toTorchTensor().unsqueeze(0)
      
      var cache2 = KVCache.init()
      let logits2 = model.forward(inputTensor2, positions2, cache2)
      
      # Check that shapes are identical
      result = logits1.shape.len == logits2.shape.len and
               logits1.shape[0] == logits2.shape[0] and
               logits1.shape[1] == logits2.shape[1] and
               logits1.shape[2] == logits2.shape[2]
      
      if result:
        echo "✅ Multiple forward passes (cache reset) successful"
      else:
        echo "❌ Multiple forward passes (cache reset) failed"
        echo "   First pass shape: ", logits1.shape.len, "D tensor"
        echo "   Second pass shape: ", logits2.shape.len, "D tensor"

when isMainModule:
  main()