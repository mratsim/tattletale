import workspace/libtorch/src/[torch_tensors, torch_tensors_sugar, c10, neural_nets]
export torch_tensors, torch_tensors_sugar, c10, neural_nets

# TODO: for now we expose C++ tuples `get`
from workspace/libtorch/src/std_cpp import get
export std_cpp.get