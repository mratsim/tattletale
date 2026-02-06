import numpy as np
from safetensors.numpy import save_file
import os

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "fixtures",
)


def make_pattern_tensor(dtype, shape, pattern_type):
    if pattern_type == "gradient":
        arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    elif pattern_type == "alternating":
        arr = (np.arange(np.prod(shape)) % 2).astype(dtype).reshape(shape)
    elif pattern_type == "repeating":
        arr = (np.arange(np.prod(shape)) % 10 + 1).astype(dtype).reshape(shape)
    elif pattern_type == "vandermonde":
        arr = np.zeros(shape, dtype=dtype)
        for idx in np.ndindex(shape):
            val = 1
            for dim_idx, coord in enumerate(idx):
                val *= (coord + 1) ** dim_idx
            arr[idx] = val

    return arr


DTYPES = {
    "F64": (np.float64, "f64"),
    "F32": (np.float32, "f32"),
    "F16": (np.float16, "f16"),
    "I64": (np.int64, "i64"),
    "I32": (np.int32, "i32"),
    "I16": (np.int16, "i16"),
    "I8": (np.int8, "i8"),
    "U64": (np.uint64, "u64"),
    "U32": (np.uint32, "u32"),
    "U16": (np.uint16, "u16"),
    "U8": (np.uint8, "u8"),
}

PATTERNS = ["gradient", "alternating", "repeating", "vandermonde"]

SHAPES = [
    (8,),
    (4, 4),
    (2, 3, 4),
    (3, 2, 2, 2),
]


def main():
    fixtures = {}

    for dtype_name, (np_dtype, ext) in DTYPES.items():
        for pattern in PATTERNS:
            for shape in SHAPES:
                key = f"{dtype_name}_{pattern}_{'x'.join(map(str, shape))}"
                arr = make_pattern_tensor(np_dtype, shape, pattern)
                fixtures[key] = arr

    save_file(fixtures, os.path.join(FIXTURES_DIR, "fixtures.safetensors"))
    print(f"Saved {len(fixtures)} tensors to {FIXTURES_DIR}/fixtures.safetensors")


if __name__ == "__main__":
    main()
