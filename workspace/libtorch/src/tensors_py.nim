# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Nim ↔ Python Tensor Bridge
## Bidirectional conversion between Nim Tensor (workspace/libtorch) and Python torch.Tensor
## objects. Both directions create independent ownership via shared refcount increment.
## Python → Nim reads THPVariable->cdata at offset 16 as a TorchTensor (intrusive_ptr copy).
## Nim → Python extracts TensorImpl* via unsafeGetTensorImpl() and wraps it in a PyCapsule.
## Both directions use {.nodestroy.} to prevent double-decrement of the intrusive refcount.

import
  nimpy,
  nimpy/py_lib as pyl,
  nimpy/py_types,
  std/importutils,
  ./raw_libtorch as F,
  ./tensors {.all.}

privateAccess(Tensor)

# =============================================================================
# THPVariable — Memory layout of Python torch.Tensor (64-bit)
# =============================================================================

type THPVariable* = object
  ## Memory layout of Python torch.Tensor (verified at offset 16 on 64-bit).
  ob_refcnt: csize_t                          # 0..7
  ob_type: pointer                            # 8..15
  cdata: TorchTensor                          # 16..23 ← at::Tensor (intrusive_ptr, 8 bytes)
  backward_hooks: pointer                     # 24..31
  post_accumulate_grad_hooks: pointer         # 32..39


const THPVariableSize* = 40  # bytes on 64-bit
# =============================================================================
# Helper: cast PyObject → ptr THPVariable
# =============================================================================

proc thpVariableFromPyObject*(pyobj: PyObject): ptr THPVariable {.inline.} =
  ## Cast a Python torch.Tensor to its underlying THPVariable struct pointer.
  ## Caller must verify the object is a torch.Tensor first.
  cast[ptr THPVariable](pyobj.privateRawPyObj())

# =============================================================================
# Helper: capsule wrappers
# =============================================================================

proc capsuleNew(data: pointer, name: cstring): PyObject =
  ## Create a PyCapsule holding ``data`` with a NULL destructor.
  ## The capsule does NOT own the pointer — the caller manages lifetime.
  let raw = pyl.pyLib.PyCapsule_New(data, name, nil)
  if raw.isNil:
    raise newException(ValueError, "PyCapsule_New failed")
  # PyCapsule_New returns a new ref; pyValueToNim wraps and manages GC lifetime.
  pyValueToNim(raw, result)

proc capsuleGetPointer(obj: PyObject, name: cstring): pointer {.inline.} =
  ## Retrieve the pointer from a PyCapsule.
  pyl.pyLib.PyCapsule_GetPointer(obj.privateRawPyObj(), name)

# =============================================================================
# Validation helpers
# =============================================================================

proc isTorchTensor*(pyobj: PyObject): bool {.inline.} =
  ## Check if a PyObject is a torch.Tensor instance.
  let tensorType = pyImport("torch").getAttr("Tensor")
  let objPtr = pyobj.privateRawPyObj()
  let objType = cast[ptr PyObjectObj](objPtr).ob_type
  result = pyl.pyLib.PyType_IsSubtype(
    cast[PyTypeObject](objType), cast[PyTypeObject](tensorType.privateRawPyObj())
  ) == 1

proc checkTorchTensor*(pyobj: PyObject) =
  ## Raise ValueError if pyobj is not a torch.Tensor.
  if not isTorchTensor(pyobj):
    let objPtr = pyobj.privateRawPyObj()
    let typ = cast[ptr PyObjectObj](objPtr).ob_type
    let typName = cast[ptr PyObjectObj](typ).ob_type
    raise newException(ValueError, "Expected torch.Tensor, got <other type>")

# =============================================================================
# Python → Nim
# =============================================================================

proc tensorFromPyObject*(pyobj: PyObject): Tensor {.nodestroy.} =
  ## Extract a Nim Tensor from a Python torch.Tensor.
  ## Shared ownership: both Python and Nim tensors can outlive each other.
  ## Raises ValueError if pyobj is not a torch.Tensor.

  # {.nodestroy.} is mandatory: TorchTensor has C++ automatic destructors.
  # Without nodestroy, Nim inserts explicit =destroy on top of C++ destructors
  # → double refcount decrement → SIGSEGV.
  # See wrapTorchTensorImpl (tensors.nim) and scaled_dot_product_attention
  # (tensors_nn.nim) for the established pattern.

  # Layout validation (runtime to avoid global/static block on importcpp types).
  assert sizeof(THPVariable) == 40, "THPVariable layout mismatch"
  assert sizeof(THPVariable.cdata) == 8, "at::Tensor is just an intrusive_ptr (8 bytes)"
  checkTorchTensor(pyobj)

  # Read THPVariable->cdata at offset 16.
  let tv = thpVariableFromPyObject(pyobj)

  # Verify cdata is not null.
  # TorchTensor is bycopy with noInit — we read it directly from memory.
  # The C++ copy constructor (triggered by `var raw = tv.cdata`) increments refcount.
  var raw = tv.cdata
  if not F.isDefined(raw):
    raise newException(ValueError, "Tensor cdata is null/undefined")

  # Wrap the TorchTensor into a Nim Tensor ref.
  # Use `move` to avoid an extra copy (extra refcount bump).
  result = wrapTorchTensorImpl(move raw)

# =============================================================================
# Nim → Python
# =============================================================================

proc tensorToPyObject*(t: Tensor): PyObject {.nodestroy.} =
  ## Wrap a Nim Tensor as a Python torch.Tensor.
  ## Shared ownership: both Python and Nim tensors can outlive each other.
  ## Returns Py_None if the Nim tensor is nil or undefined.

  # {.nodestroy.} is mandatory: same rationale as tensorFromPyObject.
  # The Tensor parameter contains an embedded TorchTensor. Nim would insert
  # redundant =destroy calls on top of C++ destructors.

  # Handle nil / undefined tensors.
  if not t.isDefined():
    pyValueToNim(pyl.pyLib.Py_None, result)
    return

  # Step 1: Get non-owning TensorImpl*.
  let implPtr = t.raw.unsafeGetTensorImpl()

  # Step 2: Create capsule with NULL destructor.
  # _wrap_tensor_impl increments refcount internally, so the capsule
  # destructor must be NULL to avoid double-free.
  let capsuleRaw = pyl.pyLib.PyCapsule_New(implPtr, "torch::autograd::AutogradEdge".cstring, nil)
  if capsuleRaw.isNil:
    raise newException(ValueError, "PyCapsule_New failed")

  # Step 3: Call torch._C._wrap_tensor_impl(capsule) → Python tensor.
  let torchC = pyImport("torch._C")
  var capsuleObj: PyObject
  pyValueToNim(capsuleRaw, capsuleObj)
  result = callMethod(torchC, "_wrap_tensor_impl", capsuleObj)
