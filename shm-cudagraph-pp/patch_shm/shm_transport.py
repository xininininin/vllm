"""Same-host, cross-vendor PP data plane for vLLM 0.27.1, PP=2/TP=1.

The scheduler and Worker.execute_model are never replaced.  A native SPSC
control library provides release/acquire publication and cross-process futexes.
Only payload pages, not control pages, are registered with CUDA/HIP.
"""

import atexit
import os
import pickle
import re
import sys
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch
from torch.profiler import record_function as _record_function

from vllm.distributed import parallel_state as ps
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu import pp_utils as pp

from shm_native import Region
from raw_dma import DMA
from descriptor_checks import validate_descriptors

_SESSION = os.environ.get("GAN_SHM_SESSION", "")
if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", _SESSION):
    raise RuntimeError("set a unique GAN_SHM_SESSION on both ranks")
_FUTEX = os.environ.get("GAN_SHM_USE_FUTEX", "1") == "1"
_D2D = os.environ.get("GAN_SHM_D2D_STAGE", "1") == "1"
_EARLY_META = os.environ.get("GAN_SHM_EARLY_META", "0") == "1"
_RAW_DMA = os.environ.get("GAN_SHM_RAW_DMA", "1") == "1"
_PACKED_RECV = os.environ.get("GAN_SHM_PACKED_RECV", "1") == "1"
_NATIVE_PUBLISHER = os.environ.get("GAN_SHM_NATIVE_PUBLISHER", "1") == "1"
_FUSED = os.environ.get("GAN_SHM_FUSED", "1") == "1"
# vLLM captures graphs independently on every PP rank, then immediately runs
# PP warmup traffic.  Heterogeneous GPUs can finish capture far apart; do not
# let the faster rank publish warmup packets before its peer can consume them.
_CUDAGRAPH_BARRIER = os.environ.get("GAN_SHM_CUDAGRAPH_BARRIER", "1") == "1"
if _NATIVE_PUBLISHER and not _RAW_DMA:
    raise ValueError("native publisher requires the raw DMA backend")
_TIMEOUT = int(os.environ.get("GAN_SHM_TIMEOUT_MS", "60000"))
if _TIMEOUT < 1000:
    raise ValueError("GAN_SHM_TIMEOUT_MS must be >= 1000")

_NOOP_RECORD = nullcontext()
record_function = (_record_function if os.environ.get("GAN_SHM_PROFILE") == "1"
                   else lambda _name: _NOOP_RECORD)

def _pp_range(direction, phase):
    """A profile range whose name identifies the logical PP channel.

    The same receiver implementation is used for rank1's forward activations
    and rank0's sampled-output feedback. Keeping the direction in every range
    is essential when traces from both ranks are viewed together.
    """
    return record_function(f"CROSS_PP::{direction.upper()}::{phase}")

# Cross-vendor process groups still need Gloo for rendezvous and non-PP work.
current_platform.dist_backend = "gloo"
type(current_platform).dist_backend = "gloo"
_original_group_init = ps.GroupCoordinator.__init__

def _group_init(self, group_ranks, local_rank, torch_distributed_backend,
                use_device_communicator, **kwargs):
    return _original_group_init(self, group_ranks, local_rank, "gloo", False,
                                **kwargs)

ps.GroupCoordinator.__init__ = _group_init

# `GPUWorker.compile_or_warm_up_model` calls `capture_model()` immediately
# before `warmup_kernels()`.  The latter exercises real PP send/recv traffic.
# Synchronize through the CPU/Gloo PP group after a successful capture so the
# initial SPSC sequence starts on both ranks together.  This wrapper is not
# reached under --enforce-eager, where vLLM skips capture_model entirely.
_original_capture_model = GPUModelRunner.capture_model

def _capture_model_with_pp_barrier(self):
    result = _original_capture_model(self)
    if _CUDAGRAPH_BARRIER:
        group = ps.get_pp_group()
        if group.world_size > 1:
            with record_function("CROSS_PP::CUDAGRAPH_CAPTURE_BARRIER"):
                torch.distributed.barrier(group=group.cpu_group)
    return result

GPUModelRunner.capture_model = _capture_model_with_pp_barrier

_channels = {}
_senders = {}
_receivers = {}
_lock = threading.Lock()

@lru_cache(maxsize=32)
def _dtype_bytes(dtype):
    return torch.empty((), dtype=dtype).element_size()

def _wire_dtype_bytes(name):
    if not name.startswith("torch."):
        raise ValueError("invalid SHM dtype name")
    dtype = getattr(torch, name[6:], None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError("unknown SHM dtype")
    return _dtype_bytes(dtype)

class _Channel:
    def __init__(self, direction):
        self.region = Region(_SESSION, direction, _FUTEX, _TIMEOUT)
        self.runtime = torch.cuda.cudart()
        self.addresses = [self.region.payload_address(i)
                          for i in range(self.region.slots)]
        self.registered = []
        self.buffers = []
        self.views = []
        try:
            for address in self.addresses:
                result = self.runtime.cudaHostRegister(
                    address, self.region.payload_span, 0)
                if int(result) != 0:
                    raise RuntimeError(
                        f"cudaHostRegister failed on {direction}: {result}")
                self.registered.append(address)
            # A slot's shared, registered host pages never move.  Reuse these
            # views instead of allocating a new PyTorch wrapper every step.
            self.buffers = [self.region.payload_buffer(i)
                            for i in range(self.region.slots)]
            self.views = [torch.frombuffer(buffer, dtype=torch.uint8)
                          for buffer in self.buffers]
        except BaseException:
            self.close()
            raise

    def close(self):
        self.views = []
        self.buffers = []
        while self.registered:
            address = self.registered.pop()
            result = self.runtime.cudaHostUnregister(address)
            if int(result) != 0:
                print(f"cudaHostUnregister failed: {result}", file=sys.stderr)
        if self.region.pointer:
            self.region.close()

def _channel(direction):
    with _lock:
        channel = _channels.get(direction)
        if channel is None:
            channel = _Channel(direction)
            _channels[direction] = channel
        return channel

class _CompletedHandle:
    def wait(self):
        return None

    def is_completed(self):
        return True

class _SourceSafeHandle:
    """The original source can be reused after D2D staging (or D2H).

    Work.wait inserts a stream dependency instead of waiting for the host
    publisher.  This retains the native Work-style source-lifetime contract
    without patching Worker.execute_model or serializing the PP pipeline.
    """

    def __init__(self, sender, event):
        self.sender = sender
        self.event = event

    def wait(self):
        self.sender.check_error()
        torch.cuda.current_stream(self.sender.device).wait_event(self.event)

    def is_completed(self):
        return self.event.query()

class _FusedSourceHandle:
    def __init__(self, sender, slot, generation):
        self.sender, self.slot, self.generation = sender, slot, generation

    def wait(self):
        self.sender.check_error()
        try:
            self.sender.dma.source_wait(self.slot, self.generation,
                                       torch.cuda.current_stream(self.sender.device))
        except BaseException:
            self.sender.region.fail()
            raise

    def is_completed(self):
        return self.sender.dma.source_query(self.slot, self.generation)

class _Sender:
    def __init__(self, direction):
        self.channel = _channel(direction)
        self.region = self.channel.region
        self.direction = direction
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dma = DMA(self.region.slots, direction) if _RAW_DMA else None
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.publisher = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="ppshm-publish")
        self.sequence = 0
        self.last_future = None
        self.staging = [None] * self.region.slots
        # Retain source allocations until the corresponding slot returns free.
        # The native publisher does not touch Python objects or acquire the GIL.
        self.inflight = [None] * self.region.slots

    def check_error(self):
        if self.last_future is not None and self.last_future.done():
            self.last_future.result()
        if self.region.error():
            raise RuntimeError(f"SHM {self.direction} peer reported failure")

    def send(self, tensors, metadata):
        self.check_error()
        descriptors = []
        total = 0
        for tensor in tensors:
            size = tensor.numel() * tensor.element_size()
            descriptors.append((tuple(tensor.shape), str(tensor.dtype),
                                total, size, tensor.is_cuda))
            total += size
        if total > self.region.payload_capacity:
            self.region.fail()
            raise ValueError(f"{self.direction} payload {total} exceeds "
                             f"preallocated {self.region.payload_capacity}")
        raw = pickle.dumps((metadata, descriptors), protocol=pickle.HIGHEST_PROTOCOL)
        if len(raw) > self.region.metadata_capacity:
            self.region.fail()
            raise ValueError(f"{self.direction} metadata exceeds preallocated slot")

        with _pp_range(self.direction, "FREE_QUEUE_WAIT"):
            slot = self.region.pop(False)
        self.inflight[slot] = None
        self.sequence += 1
        sequence = self.sequence
        host = self.channel.buffers[slot]
        sources = []
        cpu_parts = []
        event = None
        source_safe = None
        try:
            has_gpu = any(t.numel() and t.is_cuda for t in tensors)
            if has_gpu and _FUSED and _D2D and _RAW_DMA and _NATIVE_PUBLISHER:
                if _EARLY_META:
                    raise ValueError("fused submission does not enable speculative metadata")
                if self.staging[slot] is None:
                    self.staging[slot] = torch.empty(
                        self.region.payload_capacity, dtype=torch.uint8, device=self.device)
                stage = self.staging[slot]
                first, second = [], []
                for tensor, (_, _, offset, size, gpu) in zip(tensors, descriptors):
                    if not size:
                        continue
                    if gpu:
                        source = tensor.contiguous()
                        sources.append(source)
                        first.append((stage.data_ptr() + offset, source.data_ptr(), size, 3))
                        second.append((self.channel.addresses[slot] + offset,
                                       stage.data_ptr() + offset, size, 2))
                    else:
                        data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                        host[offset:offset + size] = data
                if all(gpu for _, _, _, size, gpu in descriptors if size):
                    second = [(self.channel.addresses[slot], stage.data_ptr(), total, 2)]
                self.region.write(slot, sequence, raw, total)
                self.inflight[slot] = (tensors, sources, host)
                with _pp_range(self.direction, "FUSED_DMA_SUBMIT"):
                    self.dma.fused(self.region, slot, sequence, first, second,
                                   torch.cuda.current_stream(self.device), self.copy_stream)
                return _FusedSourceHandle(self, slot, sequence)
            if has_gpu:
                stream = torch.cuda.current_stream(self.device)
                if _D2D:
                    if self.staging[slot] is None:
                        self.staging[slot] = torch.empty(
                            self.region.payload_capacity, dtype=torch.uint8,
                            device=self.device)
                    stage = self.staging[slot]
                    with _pp_range(self.direction, "D2D_STAGE"):
                        copies = []
                        for tensor, (_, _, offset, size, gpu) in zip(tensors, descriptors):
                            if size and gpu:
                                source = tensor.contiguous()
                                if self.dma:
                                    copies.append((stage.data_ptr() + offset,
                                                   source.data_ptr(), size, 3))
                                else:
                                    stage[offset:offset + size].copy_(
                                        source.view(torch.uint8).reshape(-1),
                                        non_blocking=True)
                                sources.append(source)
                        if self.dma:
                            self.dma.batch(copies, stream)
                        source_safe = stream.record_event()
                    self.copy_stream.wait_event(source_safe)
                    with torch.cuda.stream(self.copy_stream), \
                         _pp_range(self.direction, "ENQUEUE_D2H"):
                        if self.dma:
                            if _NATIVE_PUBLISHER and any(size and not gpu
                                                 for _, _, _, size, gpu in descriptors):
                                copies = [(self.channel.addresses[slot] + offset,
                                           stage.data_ptr() + offset, size, 2)
                                          for _, _, offset, size, gpu in descriptors
                                          if size and gpu]
                            else:
                                copies = [(self.channel.addresses[slot],
                                           stage.data_ptr(), total, 2)]
                            self.dma.batch(copies,
                                           self.copy_stream)
                        else:
                            target = self.channel.views[slot][:total]
                            target.copy_(stage[:total], non_blocking=True)
                        # The C publisher records a preallocated slot event.
                        # Only the D2D source-safety event needs a Python handle.
                        if not _NATIVE_PUBLISHER:
                            event = self.copy_stream.record_event()
                else:
                    self.copy_stream.wait_stream(stream)
                    with torch.cuda.stream(self.copy_stream), \
                         _pp_range(self.direction, "ENQUEUE_D2H"):
                        for tensor, (_, _, offset, size, gpu) in zip(tensors, descriptors):
                            if size and gpu:
                                source = tensor.contiguous()
                                source.record_stream(self.copy_stream)
                                if self.dma:
                                    self.dma.batch([(self.channel.addresses[slot] + offset,
                                                     source.data_ptr(), size, 2)],
                                                   self.copy_stream)
                                else:
                                    target = self.channel.views[slot][offset:offset + size]
                                    target.copy_(source.view(torch.uint8).reshape(-1),
                                                 non_blocking=True)
                                sources.append(source)
                        event = self.copy_stream.record_event()
                    source_safe = event
            # A whole-slot GPU D2H may cover CPU tensor offsets.  Keep those
            # bytes aside and fill them only after the DMA event has completed.
            for tensor, (_, _, offset, size, gpu) in zip(tensors, descriptors):
                if size and not gpu:
                    data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    cpu_parts.append((offset, data))
            self.region.write(slot, sequence, raw, total)
            if _EARLY_META:
                self.region.push(2, slot)

            if _NATIVE_PUBLISHER:
                for offset, data in cpu_parts:
                    host[offset:offset + len(data)] = data
                self.inflight[slot] = (tensors, sources, host)
                # Even an empty/CPU-only message follows previous DMA on this
                # channel's copy stream, preserving sequence publication order.
                self.dma.publish(self.region, slot, True, self.copy_stream)
                return (_SourceSafeHandle(self, source_safe) if source_safe
                        else _CompletedHandle())

            def publish():
                try:
                    _ = sources, host, tensors
                    if event is not None:
                        with _pp_range(self.direction, "D2H_WAIT"):
                            event.synchronize()
                    for offset, data in cpu_parts:
                        host[offset:offset + len(data)] = data
                    with _pp_range(self.direction, "READY_QUEUE_PUSH"):
                        self.region.push(True, slot)
                except BaseException:
                    self.region.fail()
                    raise

            self.last_future = self.publisher.submit(publish)
            return _SourceSafeHandle(self, source_safe) if source_safe else _CompletedHandle()
        except BaseException:
            self.region.fail()
            raise

    def close(self):
        self.publisher.shutdown(wait=True)
        self.check_error()
        if self.dma:
            self.dma.drain()
            self.dma.close()
        self.inflight = []

class _Receiver:
    def __init__(self, direction):
        self.channel = _channel(direction)
        self.region = self.channel.region
        self.direction = direction
        self.expected = 0
        self.dma = DMA(self.region.slots, direction) if _RAW_DMA else None
        # Exactly one publisher into free_queue even when H2D events complete
        # on different GPU streams.
        self.reclaimer = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="ppshm-reclaim")

    def receive(self):
        if _EARLY_META:
            with _pp_range(self.direction, "METADATA_QUEUE_WAIT"):
                slot = self.region.pop(2)
        else:
            with _pp_range(self.direction, "READY_QUEUE_WAIT"):
                slot = self.region.pop(True)
        sequence, raw, length = self.region.read(slot)
        self.expected += 1
        if sequence != self.expected:
            self.region.fail()
            raise RuntimeError(f"{self.direction} sequence {sequence}; "
                               f"expected {self.expected}")
        try:
            metadata, descriptors = pickle.loads(raw)
        except BaseException:
            self.region.fail()
            raise
        try:
            validate_descriptors(descriptors, length,
                                 self.region.payload_capacity, _wire_dtype_bytes)
        except BaseException:
            self.region.fail()
            raise
        return slot, metadata, descriptors, length

    def copy_to(self, slot, targets, descriptors, length, stream=None, need_event=True):
        if len(targets) != len(descriptors):
            self.region.fail()
            raise RuntimeError("SHM tensor count mismatch")
        host = self.channel.buffers[slot]
        gpu_sources = []
        try:
            if _EARLY_META:
                with _pp_range(self.direction, "READY_QUEUE_WAIT"):
                    ready_slot = self.region.pop(True)
                if ready_slot != slot:
                    raise RuntimeError(f"SHM ready slot {ready_slot}; "
                                       f"metadata slot {slot}")
            copies = []
            gpu_targets = []
            for target, (shape, dtype, offset, size, gpu) in zip(targets, descriptors):
                if not target.is_contiguous() or tuple(target.shape) != tuple(shape) or str(target.dtype) != dtype or \
                        target.numel() * target.element_size() != size or \
                        bool(target.is_cuda) != bool(gpu) or offset + size > length:
                    raise RuntimeError("SHM tensor descriptor mismatch")
                if size:
                    if self.dma and target.is_cuda:
                        gpu_targets.append(target)
                        copies.append((target.data_ptr(),
                                       self.channel.addresses[slot] + offset,
                                       size, 1))
                        target.record_stream(stream)
                    else:
                        source = self.channel.views[slot][offset:offset + size]
                        target.reshape(-1).view(torch.uint8).copy_(
                            source, non_blocking=target.is_cuda)
                        if target.is_cuda:
                            gpu_sources.append(source)
            if copies:
                # Consecutive views of the packed GPU allocation need one DMA.
                contiguous = all(copies[i][0] + copies[i][2] == copies[i + 1][0]
                                 and copies[i][1] + copies[i][2] == copies[i + 1][1]
                                 for i in range(len(copies) - 1))
                if contiguous and len({target.untyped_storage()._cdata
                                       for target in gpu_targets}) == 1:
                    copies = [(copies[0][0], copies[0][1],
                               sum(copy[2] for copy in copies), 1)]
                with _pp_range(self.direction, "H2D_SUBMIT"):
                    self.dma.batch(copies, stream)
            event = (stream.record_event() if stream is not None and
                     (need_event or not _NATIVE_PUBLISHER) else None)

            if _NATIVE_PUBLISHER:
                if stream is not None:
                    # The native publisher records an event in this stream and
                    # returns the slot to free_queue only after that event.
                    with _pp_range(self.direction, "FREE_EVENT_SUBMIT"):
                        self.dma.publish(self.region, slot, False, stream)
                else:
                    with _pp_range(self.direction, "FREE_QUEUE_PUSH_CPU"):
                        self.dma.release_cpu(self.region, slot)
                return event

            def release():
                try:
                    _ = host, gpu_sources
                    if event is not None:
                        event.synchronize()
                    with _pp_range(self.direction, "FREE_QUEUE_PUSH"):
                        self.region.push(False, slot)
                except BaseException:
                    self.region.fail()
                    raise

            self.reclaimer.submit(release)
            return event
        except BaseException:
            self.region.fail()
            raise

    def close(self):
        self.reclaimer.shutdown(wait=True)
        if self.dma:
            self.dma.drain()
            self.dma.close()

def _sender(direction):
    sender = _senders.get(direction)
    if sender is None:
        sender = _Sender(direction)
        _senders[direction] = sender
    return sender

def _receiver(direction):
    receiver = _receivers.get(direction)
    if receiver is None:
        receiver = _Receiver(direction)
        _receivers[direction] = receiver
    return receiver

def _isend_dict(self, tensor_dict, dst=None, all_gather_group=None,
                all_gather_tensors=None):
    if self.world_size == 1:
        return []
    if self.world_size != 2 or self.rank_in_group != 0 or \
            (all_gather_group is not None and all_gather_group.world_size != 1):
        raise NotImplementedError("SHM activation path requires PP=2, TP=1, rank0")
    dst = 1 if dst is None else dst
    if dst != 1:
        raise ValueError("unexpected PP activation destination")
    metadata, tensors = ps._split_tensor_dict(tensor_dict)
    return [_CompletedHandle(), _sender("forward").send(tensors, metadata)]

def _irecv_dict(self, src=None, all_gather_group=None,
                all_gather_tensors=None):
    if self.world_size == 1:
        return None, [], []
    if self.world_size != 2 or self.rank_in_group != 1 or \
            (all_gather_group is not None and all_gather_group.world_size != 1):
        raise NotImplementedError("SHM activation path requires PP=2, TP=1, rank1")
    src = 0 if src is None else src
    if src != 0:
        raise ValueError("unexpected PP activation source")
    receiver = _receiver("forward")
    slot, metadata, descriptors, length = receiver.receive()
    result = {}
    targets = []
    tensor_meta = [value for _, value in metadata
                   if isinstance(value, ps.TensorMetadata)]
    packed = None
    if _PACKED_RECV and tensor_meta and len(tensor_meta) == len(descriptors) and \
            all(item[4] for item in descriptors) and \
            len({str(value.device) for value in tensor_meta}) == 1 and \
            all(item[2] % _dtype_bytes(value.dtype) == 0
                for value, item in zip(tensor_meta, descriptors)):
        packed = torch.empty(length, dtype=torch.uint8, device=tensor_meta[0].device)
    for key, value in metadata:
        if isinstance(value, ps.TensorMetadata):
            if packed is not None:
                _, _, offset, size, _ = descriptors[len(targets)]
                target = packed[offset:offset + size].view(value.dtype).reshape(value.size)
            else:
                target = torch.empty(value.size, dtype=value.dtype, device=value.device)
            result[key] = target
            targets.append(target)
        else:
            result[key] = value
    if len(targets) != len(descriptors):
        receiver.region.fail()
        raise RuntimeError("SHM tensor metadata count mismatch")

    def consume():
        gpu = next((t for t in targets if t.is_cuda), None)
        stream = torch.cuda.current_stream(gpu.device) if gpu is not None else None
        with record_function("CROSS_PP::H2D"):
            receiver.copy_to(slot, targets, descriptors, length, stream,
                             need_event=False)

    return result, [], [consume]

ps.GroupCoordinator.isend_tensor_dict = _isend_dict
ps.GroupCoordinator.irecv_tensor_dict = _irecv_dict

_original_previous = pp.PPHandler.get_prev_sampled_outputs
_feedback_executor = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix="ppshm-feedback")

def _receive_feedback(self, batch):
    assert not self.is_last_rank
    mask = pp.compute_need_sampled_mask(batch)
    if mask is None:
        return False
    n = batch.num_reqs
    generation = self.req_idx_gen_np[batch.idx_mapping_np]
    if _PACKED_RECV:
        token_bytes = n * self.max_sample_len * 8
        buffer = torch.empty(token_bytes + 2 * n * 4, dtype=torch.uint8,
                             device=self.device)
        tokens = buffer[:token_bytes].view(torch.int64).reshape(n, self.max_sample_len)
        counts = buffer[token_bytes:].view(torch.int32).reshape(2, n)
    else:
        tokens = torch.empty((n, self.max_sample_len), dtype=torch.int64,
                             device=self.device)
        counts = torch.empty((2, n), dtype=torch.int32, device=self.device)
    ready = self.main_stream.record_event()
    pending = pp.PendingRecv(None, tokens, counts[0], counts[1],
                             batch.idx_mapping, batch.idx_mapping_np,
                             mask, generation)

    def finish():
        receiver = _receiver("feedback")
        slot, metadata, descriptors, length = receiver.receive()
        if metadata != "sampled-output":
            receiver.region.fail()
            raise RuntimeError("unexpected SHM feedback metadata")
        with torch.inference_mode(), torch.cuda.device(self.device), \
             torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_event(ready)
            with record_function("CROSS_PP::FEEDBACK_H2D"):
                event = receiver.copy_to(slot, [tokens, counts],
                                         descriptors, length,
                                         self.broadcast_stream)
            tokens.record_stream(self.broadcast_stream)
            counts.record_stream(self.broadcast_stream)
        return event

    pending.gan_shm_future = _feedback_executor.submit(finish)
    self.queue[-1] = pending
    return bool(mask.all())

def _previous_feedback(self):
    pending = self.queue[0] if self.queue else None
    if pending is not None and hasattr(pending, "gan_shm_future"):
        # Separates a genuine host dependency on the feedback from the
        # asynchronous READY_QUEUE_WAIT in the feedback worker thread.
        with _pp_range("feedback", "PENDING_FUTURE_WAIT"):
            pending.event = pending.gan_shm_future.result()
    return _original_previous(self)

def _broadcast_feedback(self, sampled_token_ids, num_sampled,
                        num_rejected, batch):
    assert self.is_last_rank
    if pp.compute_need_sampled_mask(batch) is None:
        return
    counts = torch.stack((num_sampled, num_rejected), dim=0)
    _sender("feedback").send((sampled_token_ids, counts), "sampled-output")

pp.PPHandler.receive = _receive_feedback
pp.PPHandler.get_prev_sampled_outputs = _previous_feedback
pp.PPHandler.broadcast = _broadcast_feedback

def _shutdown():
    unsafe = False
    _feedback_executor.shutdown(wait=True)
    for sender in _senders.values():
        try:
            sender.close()
        except BaseException as exc:
            unsafe = True
            print(f"SHM publisher shutdown: {exc}", file=sys.stderr)
    for receiver in _receivers.values():
        try:
            receiver.close()
        except BaseException as exc:
            unsafe = True
            print(f"SHM reclaimer shutdown: {exc}", file=sys.stderr)
    if unsafe:
        print("Keeping registered SHM mapped until process exit after failed drain",
              file=sys.stderr)
        return
    for channel in _channels.values():
        try:
            channel.close()
        except BaseException as exc:
            print(f"SHM channel shutdown: {exc}", file=sys.stderr)

atexit.register(_shutdown)
print("GAN_SHM_SPSC transport loaded", file=sys.stderr, flush=True)

