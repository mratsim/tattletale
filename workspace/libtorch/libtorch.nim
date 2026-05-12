import workspace/libtorch/src/tensors
export tensors

# An important note. There are 3 types of equality:
#
# Referential equality, are the tensor objects the same instance.
# - For Nim, this is through `==`. This is used if a Tensor is stored in a table.
# - For libtorch, you have `is_same` and `is_alias_of`.
#   2 tensors may have different Nim reference but same libtorch reference
#   if the second one is created using `to` but without enforcing copy
#   and copy is deemed unnecessary (same device, same type)
#
# Then you have value equality, do 2 tensors hold the same values.
# - This is done through `equal` -> bool
#
# And you have elementwise equality, this returns a true/false tensor
# - This is done through `eq`