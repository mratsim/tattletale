import workspace/libtorch/src/[torch_tensors, torch_tensors_sugar, c10]
export torch_tensors, torch_tensors_sugar, c10

# TODO: for now we expose C++ tuples `get`
from workspace/libtorch/src/std_cpp import get
export std_cpp.get