# libtorch

This wraps low-level TorchTensor and ancillary torch types like IntArrayRef into high-level Tensor and a Nim API.

Unfortunately we can't use torch::Tensor directly as their intrusive_ptr design and expected invariant (a zero refcount cannot never become one again)
and Nim constructors/destructors/move semantics interfere with each other (Nim sequence assignment doesn't generate std::move but a copy constructor and then =wasMoved()).

This means just like Python and the C++ stable API we have double the heap allocation
and double indirection refcounting (PyObject in Python, shared_ptr in torch::stable::Tensor, ref in Nim)
and double dereference to access the Tensor data, including shape and stride smetadata :/.

At least Nim's refcounting is more efficient than C++ shared_ptr or Python as
a variable that doesn't leave it's scope will be statically tracked, and moves/sinks can optimize refcounting away.