# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Dtype and device enums for tensorcery.
##
## Dtype is the element type of tensor storage.
## DeviceKind is the physical device.
## BackendKind is the compile-time specialization key of the typed TensorImpl layer.
## Dtype and DeviceKind are runtime values on the Tensor face, BackendKind is a generic parameter.

type
  Dtype* = enum
    ## Element type of tensor storage, modeled over libtorch's ScalarKind.
    dtBool
    dtUint8
    dtInt8
    dtInt16
    dtInt32
    dtInt64
    dtFloat16
    dtFloat32
    dtFloat64

  DeviceKind* = enum
    ## Physical device hosting the Tensor.
    devCpu
    devMetal
    devVulkan

  BackendKind* = enum
    ## Compile-time specialization key of TensorImpl, one backend per kind.
    bkCpu
    bkMetal
    bkVulkan

func itemsize*(t: Dtype): int =
  ## Returns the storage size of one element in bytes.
  case t
  of dtBool, dtUint8, dtInt8: result = 1
  of dtInt16, dtFloat16: result = 2
  of dtInt32, dtFloat32: result = 4
  of dtInt64, dtFloat64: result = 8

func isFloat*(t: Dtype): bool =
  case t
  of dtFloat16, dtFloat32, dtFloat64: result = true
  of dtBool, dtUint8, dtInt8, dtInt16, dtInt32, dtInt64: result = false

func toDtype*(T: typedesc): Dtype =
  ## Returns the Dtype for a Nim element type.
  when T is bool:
    result = dtBool
  elif T is uint8:
    result = dtUint8
  elif T is int8:
    result = dtInt8
  elif T is int16:
    result = dtInt16
  elif T is int32:
    result = dtInt32
  elif T is int64:
    result = dtInt64
  elif T is int:
    result = dtInt64
  elif T is float32:
    result = dtFloat32
  elif T is float64:
    result = dtFloat64
  else:
    {.error: "no Dtype for this Nim type, tensorcery supports bool, uint8, int8, int16, int32, int64, float32, float64".}

func toDeviceKind*(bk: BackendKind): DeviceKind =
  ## Returns the runtime device of a backend specialization.
  case bk
  of bkCpu: result = devCpu
  of bkMetal: result = devMetal
  of bkVulkan: result = devVulkan
