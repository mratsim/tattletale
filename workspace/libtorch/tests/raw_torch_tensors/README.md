# Raw Torch Tensors tests

The `test_*` are validated to work with raw torch::Tensor

The `bug_test_*` are blockers that prevent using raw tensors directly and forces to rewrap them in a ref object.
For reference, in Python they are wrapped in a ref counted PyObject and in the stable C++ API they are wrapped in a shared_ptr.

All the bugs are related to assignmenent and lifetime:
- either default initialization in sequences being ambiguous - https://github.com/nim-lang/Nim/issues/25803
- or issues with copy constructor / move constructor / wasMoved