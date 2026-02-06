import numpy as np
from safetensors.numpy import save_file
import os

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "fixtures",
)


def main():
    shape = (5, 5)
    x = np.arange(1, 6, dtype=np.float64)
    vandermonde = np.vander(x, increasing=True).T

    fixtures = {
        "F64_vandermonde_5x5": vandermonde.astype(np.float64),
        "F32_vandermonde_5x5": vandermonde.astype(np.float32),
    }

    output_path = os.path.join(FIXTURES_DIR, "vandermonde.safetensors")
    save_file(fixtures, output_path)
    print(f"Saved Vandermonde tensors to {output_path}")
    print(f"Shape: {shape}")
    print(f"Vandermonde matrix (increasing powers):")
    print(vandermonde)


if __name__ == "__main__":
    main()
