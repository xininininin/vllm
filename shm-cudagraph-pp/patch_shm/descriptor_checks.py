"""Reject invalid byte layouts before allocating targets or submitting DMA."""
import math

def validate_descriptors(descriptors, length, capacity, dtype_bytes):
    if type(length) is not int or not 0 <= length <= capacity:
        raise ValueError("invalid SHM payload length")
    if not isinstance(descriptors, (list, tuple)):
        raise ValueError("invalid SHM descriptor list")
    cursor = 0
    for item in descriptors:
        if not isinstance(item, (list, tuple)) or len(item) != 5:
            raise ValueError("invalid SHM tensor descriptor")
        shape, dtype, offset, size, gpu = item
        if not isinstance(shape, (list, tuple)) or any(
                type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("invalid SHM tensor shape")
        if not isinstance(dtype, str) or type(gpu) is not bool:
            raise ValueError("invalid SHM tensor dtype/device")
        if type(offset) is not int or type(size) is not int or size < 0 or offset != cursor:
            raise ValueError("SHM offsets must form a contiguous non-overlapping layout")
        itemsize = dtype_bytes(dtype)
        if size != math.prod(shape) * itemsize or offset + size > length:
            raise ValueError("SHM shape/dtype/byte length mismatch")
        cursor += size
    if cursor != length:
        raise ValueError("SHM payload descriptor length mismatch")
