import torch
from safetensors.torch import save_file, load_file
import os

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "fixtures",
)


def main():
    shape = (5, 5)
    x = torch.arange(1, 6, dtype=torch.float64)
    vandermonde = torch.vander(x, increasing=True).T

    fixtures = {
        "F64_vandermonde_5x5": vandermonde.to(torch.float64).contiguous(),
        "F32_vandermonde_5x5": vandermonde.to(torch.float32).contiguous(),
        "BF16_vandermonde_5x5": vandermonde.to(torch.bfloat16).contiguous(),
    }

    output_path = os.path.join(FIXTURES_DIR, "vandermonde.safetensors")
    save_file(fixtures, output_path)
    print(f"Saved Vandermonde tensors to {output_path}")
    print(f"Shape: {shape}")
    print(f"Vandermonde matrix (increasing powers):")
    print(vandermonde)

    print("\nVerifying BF16 fixture reload...")
    loaded = load_file(output_path)
    bf16_original = fixtures["BF16_vandermonde_5x5"]
    bf16_loaded = loaded["BF16_vandermonde_5x5"]
    assert torch.equal(bf16_loaded, bf16_original), "BF16 reload mismatch"
    print("BF16 fixture reload verified successfully")


if __name__ == "__main__":
    main()
