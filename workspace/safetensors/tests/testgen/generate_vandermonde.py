import torch
from safetensors.torch import save_file, load_file
import os

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "fixtures",
)


def main():
    # Generate 5x5 shifted Vandermonde matrix: v[i, j] = i^(j+1)
    # [[   1    1    1    1    1]
    #  [   2    4    8   16   32]
    #  [   3    9   27   81  243]
    #  [   4   16   64  256 1024]
    #  [   5   25  125  625 3125]]
    vandermonde = torch.arange(1, 6).reshape(-1, 1) ** torch.arange(1, 6)

    fixtures = {
        "F64_vandermonde_5x5": vandermonde.to(torch.float64).contiguous(),
        "F32_vandermonde_5x5": vandermonde.to(torch.float32).contiguous(),
        "BF16_vandermonde_5x5": vandermonde.to(torch.bfloat16).contiguous(),
    }

    output_path = os.path.join(FIXTURES_DIR, "vandermonde.safetensors")
    save_file(fixtures, output_path)
    print(f"Saved Vandermonde tensors to {output_path}")
    print(f"Shape: {vandermonde.shape}")
    print(f"5x5 shifted Vandermonde matrix: v[i, j] = i^(j+1):")
    print(vandermonde)

    print("\nVerifying BF16 fixture reload...")
    loaded = load_file(output_path)
    bf16_original = fixtures["BF16_vandermonde_5x5"]
    bf16_loaded = loaded["BF16_vandermonde_5x5"]
    assert torch.equal(bf16_loaded, bf16_original), "BF16 reload mismatch"
    print("BF16 fixture reload verified successfully")


if __name__ == "__main__":
    main()
