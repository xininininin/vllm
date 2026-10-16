"""Submit contiguous CUDA/HIP copies in one C call on the caller's stream.

This changes only copy submission overhead. PyTorch owns every allocation,
stream and event; the SHM transport still transfers slot ownership after DMA.
"""
import ctypes
import ctypes.util
import glob
import os
from pathlib import Path

import torch
from shm_native import _LIB_PATH

class Copy(ctypes.Structure):
    _fields_ = [("destination", ctypes.c_uint64), ("source", ctypes.c_uint64),
                ("bytes", ctypes.c_uint64), ("kind", ctypes.c_int32),
                ("padding", ctypes.c_int32)]

class DMA:
    def __init__(self, slots=2, direction="unknown"):
        path = Path(__file__).resolve().parent.parent / "native" / "libppdma.so"
        self.lib = ctypes.CDLL(str(path))
        self.lib.pp_dma_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                                         ctypes.c_uint32, ctypes.c_int, ctypes.c_char_p]
        self.lib.pp_dma_open.restype = ctypes.c_void_p
        self.lib.pp_dma_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(Copy),
                                          ctypes.c_uint32, ctypes.c_uint64]
        self.lib.pp_dma_batch.restype = ctypes.c_int
        self.lib.pp_dma_close.argtypes = [ctypes.c_void_p]
        self.lib.pp_dma_close.restype = ctypes.c_int
        self.lib.pp_dma_publish.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_uint32, ctypes.c_int,
                                            ctypes.c_uint64]
        self.lib.pp_dma_publish.restype = ctypes.c_int
        self.lib.pp_dma_drain.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.pp_dma_drain.restype = ctypes.c_int
        self.lib.pp_dma_release_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_uint32]
        self.lib.pp_dma_release_cpu.restype = ctypes.c_int
        self.lib.pp_dma_fused.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(Copy), ctypes.c_uint32,
            ctypes.POINTER(Copy), ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.pp_dma_fused.restype = ctypes.c_int
        self.lib.pp_dma_source_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                ctypes.c_uint64, ctypes.c_uint64]
        self.lib.pp_dma_source_wait.restype = ctypes.c_int
        self.lib.pp_dma_source_query.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                 ctypes.c_uint64]
        self.lib.pp_dma_source_query.restype = ctypes.c_int
        hip = bool(torch.version.hip)
        override = os.environ.get("GAN_SHM_GPU_RUNTIME")
        if override:
            paths = [override]
        elif hip:
            paths = [ctypes.util.find_library("amdhip64"), "libamdhip64.so"]
            paths += glob.glob("/opt/rocm*/lib/libamdhip64.so*")
        else:
            # Use the same runtime already loaded by PyTorch, never a driver
            # library (libcuda). Maps may contain duplicate mappings.
            paths = list(dict.fromkeys(line.split()[-1] for line in
                Path("/proc/self/maps").read_text().splitlines()
                if "libcudart.so" in line and line.split()[-1].startswith("/")))
            paths += ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
        self.pointer = None
        for runtime in paths:
            if runtime:
                self.pointer = self.lib.pp_dma_open(os.fsencode(runtime), int(hip),
                                                     os.fsencode(_LIB_PATH), slots,
                                                     torch.cuda.current_device(),
                                                     os.fsencode(direction))
                if self.pointer:
                    break
        if not self.pointer:
            raise RuntimeError(f"cannot open {'HIP' if hip else 'CUDA'} memcpy runtime")

    def batch(self, copies, stream):
        if not copies:
            return
        commands = (Copy * len(copies))(*(Copy(*copy, 0) for copy in copies))
        result = self.lib.pp_dma_batch(self.pointer, commands, len(copies),
                                      stream.cuda_stream)
        if result:
            raise RuntimeError(f"CUDA/HIP asynchronous DMA submission failed: {result}")

    def fused(self, region, slot, generation, stage, host, model_stream, copy_stream):
        first = (Copy * len(stage))(*(Copy(*copy, 0) for copy in stage))
        second = (Copy * len(host))(*(Copy(*copy, 0) for copy in host))
        result = self.lib.pp_dma_fused(self.pointer, region.pointer, slot, generation,
                    first, len(stage), second, len(host),
                    model_stream.cuda_stream, copy_stream.cuda_stream)
        if result:
            raise RuntimeError(f"Fused CUDA/HIP DMA submission failed: {result}")

    def source_wait(self, slot, generation, stream):
        if not self.pointer:
            raise RuntimeError("source handle used after DMA runtime close")
        result = self.lib.pp_dma_source_wait(self.pointer, slot, generation,
                                             stream.cuda_stream)
        if result:
            raise RuntimeError(f"Source-safe stream dependency failed: {result}")

    def source_query(self, slot, generation):
        if not self.pointer:
            return True  # A successful close drained all DMA jobs.
        result = self.lib.pp_dma_source_query(self.pointer, slot, generation)
        if result < 0:
            raise RuntimeError(f"Source-safe GPU event query failed: {result}")
        return bool(result)

    def publish(self, region, slot, ready, stream):
        result = self.lib.pp_dma_publish(self.pointer, region.pointer, slot,
                                         int(ready), stream.cuda_stream)
        if result:
            raise RuntimeError(f"CUDA/HIP event job submission failed: {result}")

    def drain(self):
        result = self.lib.pp_dma_drain(self.pointer, 60000)
        if result:
            raise RuntimeError(f"DMA publisher jobs did not drain: {result}")

    def release_cpu(self, region, slot):
        result = self.lib.pp_dma_release_cpu(self.pointer, region.pointer, slot)
        if result:
            raise RuntimeError(f"CPU-only slot release failed: {result}")

    def close(self):
        if self.pointer:
            result = self.lib.pp_dma_close(self.pointer)
            if result:
                raise RuntimeError(f"DMA runtime could not close safely: {result}")
            self.pointer = None
