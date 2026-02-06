import numpy as np
from safetensors.numpy import save_file
import os

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "fixtures",
)


def make_pattern_tensor(dtype, shape, pattern_type):
    if dtype == np.float32:
        base = np.arange(np.prod(shape), dtype=np.float32) * 0.01
    elif dtype == np.float64:
        base = np.arange(np.prod(shape), dtype=np.float64) * 0.001
    elif dtype == np.int32:
        base = np.arange(np.prod(shape), dtype=np.int32)
    elif dtype == np.int64:
        base = np.arange(np.prod(shape), dtype=np.int64)
    elif dtype == np.int16:
        base = (np.arange(np.prod(shape), dtype=np.int32) % 32768).astype(np.int16)
    elif dtype == np.int8:
        base = (np.arange(np.prod(shape), dtype=np.int32) % 128).astype(np.int8)
    elif dtype == np.uint32:
        base = np.arange(np.prod(shape), dtype=np.uint32)
    elif dtype == np.uint64:
        base = np.arange(np.prod(shape), dtype=np.uint64)
    elif dtype == np.uint16:
        base = (np.arange(np.prod(shape), dtype=np.uint32) % 65536).astype(np.uint16)
    elif dtype == np.uint8:
        base = np.arange(np.prod(shape), dtype=np.uint8)
    elif dtype == np.bool_:
        base = (np.arange(np.prod(shape)) % 2).astype(np.bool_)
    elif dtype == np.float16:
        base = np.arange(np.prod(shape), dtype=np.float16) * 0.01
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    arr = base.reshape(shape)

    if pattern_type == "identity":
        arr = np.zeros(shape, dtype=dtype)
        for i in range(min(shape)):
            strides = np.cumprod((1,) + shape[:-1])
            idx = tuple((i * s) % shape[j] for j, s in enumerate(strides))
            arr[idx] = 1.0 if np.issubdtype(dtype, np.floating) else 1

    elif pattern_type == "checkerboard":
        arr = np.zeros(shape, dtype=dtype)
        for idx in np.ndindex(shape):
            if sum(idx) % 2 == 0:
                val = 1.0 if np.issubdtype(dtype, np.floating) else 1
                arr[idx] = val

    elif pattern_type == "gradient":
        if dtype == np.bool_:
            arr = (np.arange(np.prod(shape)) % 2).astype(dtype).reshape(shape)
        else:
            arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    elif pattern_type == "alternating":
        arr = (np.arange(np.prod(shape)) % 2).astype(dtype)
        arr = arr.reshape(shape)

    elif pattern_type == "repeating":
        arr = (np.arange(np.prod(shape)) % 10 + 1).astype(dtype)
        arr = arr.reshape(shape)

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
    "BOOL": (np.bool_, "bool"),
}

PATTERNS = ["gradient", "identity", "checkerboard", "alternating", "repeating"]

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
    print(
        f"Saved {len(fixtures)} tensors to {FIXTURES_DIR}/fixtures.safetensors"
    )


if __name__ == "__main__":
    main()
