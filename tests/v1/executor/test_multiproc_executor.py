# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import (
    STEP_RPC_METHODS,
    MultiprocExecutor,
    WorkerProc,
)
from vllm.v1.outputs import DraftTokenIds


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _FailingThenNextQueue:
    """Serves a failing step RPC, and a second RPC if the worker comes back
    for it."""

    def __init__(self) -> None:
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        assert self.dequeue_count == 1, "worker dequeued another RPC after failing"
        return "execute_model", (), {}, 0


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


def _worker_proc_stub(rank: int, worker: Any, outputs: list[Any]) -> Any:
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = rank
    worker_proc.worker = worker
    worker_proc.handle_output = outputs.append
    return worker_proc


def _fail(*args, **kwargs):
    raise RuntimeError("test error")


@pytest.mark.parametrize("method", sorted(STEP_RPC_METHODS))
@pytest.mark.parametrize("rank,is_output_rank", [(0, True), (1, False)])
def test_step_rpc_failure_terminates_worker(caplog, method, rank, is_output_rank):
    """A step RPC that raises must end the worker on every rank, not only on
    the output rank. A rank that keeps serving RPCs after a failed step enters
    the next step with its collectives out of step with its peers, which
    mispairs them and deadlocks the whole parallel group."""
    outputs: list[Any] = []
    worker_proc = _worker_proc_stub(rank, SimpleNamespace(**{method: _fail}), outputs)

    with pytest.raises(RuntimeError, match="test error"):
        worker_proc._execute_worker_rpc((method, (), {}, 0))

    # The failure must be attributable: rank and method in the message, and the
    # original traceback in the log record.
    record = caplog.records[-1]
    assert method in record.getMessage()
    assert f"rank {rank}" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError

    # The output rank still reports the failure to the engine; the non-output
    # rank has no result to report and must not pretend it succeeded.
    if is_output_rank:
        assert len(outputs) == 1
        assert isinstance(outputs[0], RuntimeError)
        assert str(outputs[0]) == "test error"
    else:
        assert outputs == []


@pytest.mark.parametrize("method", sorted(STEP_RPC_METHODS))
def test_step_rpc_success_is_unchanged(method):
    outputs: list[Any] = []
    worker_proc = _worker_proc_stub(
        0, SimpleNamespace(**{method: lambda: "step output"}), outputs
    )

    worker_proc._execute_worker_rpc((method, (), {}, 0))

    assert outputs == ["step output"]


def test_non_step_rpc_failure_does_not_terminate_worker():
    """Only step RPCs are fatal. Every other RPC keeps its existing semantics
    of reporting the failure and returning to the busy loop."""
    outputs: list[Any] = []
    worker_proc = _worker_proc_stub(
        0, SimpleNamespace(take_draft_token_ids=_fail), outputs
    )

    worker_proc._execute_worker_rpc(("take_draft_token_ids", (), {}, 0))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)


def test_worker_busy_loop_does_not_dequeue_after_step_failure():
    queue = _FailingThenNextQueue()
    outputs: list[Any] = []
    worker_proc = _worker_proc_stub(0, SimpleNamespace(execute_model=_fail), outputs)
    worker_proc.rpc_broadcast_mq = queue

    with pytest.raises(RuntimeError, match="test error"):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 1


@pytest.mark.parametrize("stalled", [False, True])
@pytest.mark.parametrize(
    "method,result",
    [
        ("take_draft_token_ids", DraftTokenIds(["req"], [[1, 2, 3]])),
        ("execute_dummy_batch", None),
    ],
)
def test_model_rpc_uses_execute_model_timeout(monkeypatch, method, result, stalled):
    """A wedged model RPC must not block EngineCore forever. For
    execute_dummy_batch this is an idle DP rank stuck in a collective with a
    hung peer."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7")

    def dequeue(*, timeout=None):
        assert timeout is not None and 0 < timeout <= 7
        if stalled:
            raise TimeoutError
        return WorkerProc.ResponseStatus.SUCCESS, result

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.output_rank = 1
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda payload: None)
    executor.response_mqs = [None, SimpleNamespace(dequeue=dequeue)]

    if stalled:
        with pytest.raises(TimeoutError, match=f"{method} timed out"):
            getattr(executor, method)()
    else:
        assert getattr(executor, method)() is result
