# Nim-C++ tips & tricks

## ambiguous overload for `operator=` with TorchTensor

The C++ compiler might return
```
error: ambiguous overload for ‘operator=’ (operand types are ‘at::Tensor’ and ‘<brace-enclosed initializer list>’)
 1145 |                                                                                                                                         expectedTensor__tests95safetensors_u670 = {};
```

This happens for procedures assigning values to a top-level variable, for example in testing.

The C++ code generated looks like this

```cpp
MyType foo = {};
foo = my_function(a, b);
```

The solution is to always wrap your tests in a function, then Nim will use

```
auto foo = my_function(a, b);
```