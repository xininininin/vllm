"""ctypes binding for the small Linux shared-memory SPSC control library."""

import ctypes
import errno
import os
from pathlib import Path

_LIB_PATH = Path(os.environ.get(
    "GAN_SHM_LIBRARY",
    str(Path(__file__).resolve().parent.parent / "native" / "libppshm.so"),
))
_lib = ctypes.CDLL(str(_LIB_PATH), use_errno=True)

_lib.pp_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_uint32]
_lib.pp_create.restype = ctypes.c_int
_lib.pp_open.argtypes = [ctypes.c_char_p]
_lib.pp_open.restype = ctypes.c_void_p
_lib.pp_close.argtypes = [ctypes.c_void_p]
_lib.pp_close.restype = ctypes.c_int
_lib.pp_unlink.argtypes = [ctypes.c_char_p]
_lib.pp_unlink.restype = ctypes.c_int
_lib.pp_queue_pop.argtypes = [ctypes.c_void_p, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                              ctypes.c_int]
_lib.pp_queue_pop.restype = ctypes.c_int
_lib.pp_queue_push.argtypes = [ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
_lib.pp_queue_push.restype = ctypes.c_int
_lib.pp_slot_write_header.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                      ctypes.c_uint64, ctypes.c_uint32,
                                      ctypes.c_uint32]
_lib.pp_slot_write_header.restype = ctypes.c_int
_lib.pp_slot_read_header.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                     ctypes.POINTER(ctypes.c_uint64),
                                     ctypes.POINTER(ctypes.c_uint32),
                                     ctypes.POINTER(ctypes.c_uint32)]
_lib.pp_slot_read_header.restype = ctypes.c_int
for _name in ("pp_slot_metadata", "pp_slot_payload"):
    _fn = getattr(_lib, _name)
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _fn.restype = ctypes.c_void_p
for _name in ("pp_slot_count", "pp_metadata_capacity", "pp_payload_capacity",
              "pp_payload_span"):
    _fn = getattr(_lib, _name)
    _fn.argtypes = [ctypes.c_void_p]
    _fn.restype = ctypes.c_uint32
_lib.pp_set_error.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.pp_set_error.restype = ctypes.c_int
_lib.pp_get_error.argtypes = [ctypes.c_void_p]
_lib.pp_get_error.restype = ctypes.c_int

def _check(result, action):
    if result < 0:
        raise OSError(-result, f"{action}: {os.strerror(-result)}")

def channel_name(session, direction):
    if not session or len(session) > 40 or not all(
        c.isascii() and (c.isalnum() or c in "_-") for c in session
    ):
        raise ValueError("GAN_SHM_SESSION must be 1-40 ASCII letters/digits/_/-")
    if direction not in ("forward", "feedback"):
        raise ValueError("unknown SHM channel")
    return f"/ganpp_{session}_{direction}".encode("ascii")

def create(session, direction, slots, metadata_bytes, payload_bytes):
    _check(_lib.pp_create(channel_name(session, direction), slots,
                          metadata_bytes, payload_bytes), "pp_create")

def unlink(session, direction):
    _check(_lib.pp_unlink(channel_name(session, direction)), "pp_unlink")

class Region:
    def __init__(self, session, direction, use_futex=True, timeout_ms=60000):
        self.name = channel_name(session, direction)
        self.pointer = _lib.pp_open(self.name)
        if not self.pointer:
            code = ctypes.get_errno()
            raise OSError(code, f"pp_open {self.name!r}: {os.strerror(code)}")
        self.slots = _lib.pp_slot_count(self.pointer)
        self.metadata_capacity = _lib.pp_metadata_capacity(self.pointer)
        self.payload_capacity = _lib.pp_payload_capacity(self.pointer)
        self.payload_span = _lib.pp_payload_span(self.pointer)
        self.use_futex = int(use_futex)
        self.timeout_ms = timeout_ms

    def pop(self, ready):
        slot = ctypes.c_uint32()
        result = _lib.pp_queue_pop(self.pointer, int(ready), ctypes.byref(slot),
                                   self.timeout_ms, self.use_futex)
        self._queue_check(result, "pop", ready)
        return slot.value

    def push(self, ready, slot):
        result = _lib.pp_queue_push(self.pointer, int(ready), slot,
                                    self.timeout_ms, self.use_futex)
        self._queue_check(result, "push", ready)

    def _queue_check(self, result, action, kind):
        if result < 0:
            # A timed-out channel is aborted, never reused as though healthy.
            # Wake the peer instead of leaving it asleep on another queue.
            if result != -errno.EPIPE:
                self.fail(-result)
            _check(result, f"{self.name.decode()} queue={int(kind)} {action}, "
                   f"timeout_ms={self.timeout_ms}, shared_error={self.error()}")

    def write(self, slot, sequence, metadata, payload_bytes):
        if len(metadata) > self.metadata_capacity:
            raise ValueError("metadata exceeds preallocated SHM slot")
        _check(_lib.pp_slot_write_header(self.pointer, slot, sequence,
                                         len(metadata), payload_bytes),
               "slot_write_header")
        ctypes.memmove(_lib.pp_slot_metadata(self.pointer, slot), metadata,
                       len(metadata))

    def read(self, slot):
        seq = ctypes.c_uint64()
        meta_len = ctypes.c_uint32()
        data_len = ctypes.c_uint32()
        _check(_lib.pp_slot_read_header(self.pointer, slot, ctypes.byref(seq),
                                        ctypes.byref(meta_len),
                                        ctypes.byref(data_len)),
               "slot_read_header")
        metadata = ctypes.string_at(_lib.pp_slot_metadata(self.pointer, slot),
                                    meta_len.value)
        return seq.value, metadata, data_len.value

    def payload_address(self, slot):
        address = _lib.pp_slot_payload(self.pointer, slot)
        if not address:
            raise ValueError("invalid payload slot")
        return address

    def payload_buffer(self, slot, length=None):
        size = self.payload_capacity if length is None else length
        if size < 0 or size > self.payload_capacity:
            raise ValueError("payload exceeds preallocated SHM slot")
        return (ctypes.c_ubyte * size).from_address(self.payload_address(slot))

    def fail(self, code=errno.EIO):
        _check(_lib.pp_set_error(self.pointer, code), "set_error")

    def error(self):
        return _lib.pp_get_error(self.pointer)

    def close(self):
        if self.pointer:
            _check(_lib.pp_close(self.pointer), "pp_close")
            self.pointer = None
