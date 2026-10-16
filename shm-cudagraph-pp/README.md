# GAN_SHM_SPSC — 同机跨厂商 vLLM PP 数据面

vLLM 0.27.1，PP=2 / TP=1。不替换 scheduler 和 `Worker.execute_model`，
由原生 SPSC 控制库提供 release/acquire 发布和跨进程 futex；
仅 payload 页注册 CUDA/HIP，控制页不注册。

## 目录结构

```
patch_shm/            # PYTHONPATH 指向这里
  sitecustomize.py    # 入口 shim：GAN_SHM_SPSC=1 时自动导入 shm_transport
  shm_transport.py    # 核心补丁：isend/irecv_dict、PPHandler feedback、融合提交
  shm_native.py       # ctypes 绑定 libppshm.so（SPSC 控制库）
  raw_dma.py          # ctypes 绑定 libppdma.so（批量 CUDA/HIP memcpyAsync）
  descriptor_checks.py# 接收侧字节布局校验
native/
  libppshm.c          # 共享内存 SPSC 队列（shm_open + futex）
  libppdma.c          # DMA 发布线程（事件同步后推队列）
  ppshm.h             # 控制库头文件
```

编译产物应为 `native/libppshm.so` 和 `native/libppdma.so`。

## 启动容器（改成你自己的）

```bash
docker run -d \
  --name linshuili-vllm-nv \
  --gpus device=0 \
  --ipc=host \
  --network=host \
  -v /data/models/Qwen2.5-3B-Instruct:/models/Qwen2.5-3B-Instruct:ro \
  -v /data/linshuili/gloo_infer:/work \
  --entrypoint /bin/bash vllm/vllm-openai:latest -lc 'sleep infinity'

docker run -d \
  --name linshuili-vllm-amd \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 992 \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  -v /data/models/Qwen2.5-3B-Instruct:/models/Qwen2.5-3B-Instruct:ro \
  -v /data/linshuili/gloo_infer:/work \
  --entrypoint /bin/bash vllm/vllm-openai-rocm:latest -lc 'sleep infinity'
```

## 启动 vLLM（启用 cudagraph）

AMD 侧（rank 1）：

```bash
docker exec -it \
  -e PYTHONPATH=/work/patch_shm \
  -e GAN_SHM_SPSC=1 \
  -e GAN_SHM_SESSION=manual-lsl \
  -e GAN_SHM_LIBRARY=/work/native/libppshm.so \
  -e GAN_SHM_USE_FUTEX=1 \
  -e GAN_SHM_D2D_STAGE=1 \
  -e GAN_SHM_EARLY_META=0 \
  -e GAN_SHM_RAW_DMA=1 \
  -e GAN_SHM_PACKED_RECV=1 \
  -e GAN_SHM_NATIVE_PUBLISHER=1 \
  -e GAN_SHM_FUSED=1 \
  -e GAN_SHM_PROFILE=0 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_HOST_IP=127.0.0.1 \
  -e VLLM_CACHE_ROOT=/work/runs/manual-1008a/cache-0 \
  -e VLLM_PP_LAYER_PARTITION=18,18 \
  -e GLOO_SOCKET_IFNAME=lo \
  -e OMP_NUM_THREADS=4 \
  -e VLLM_LOGGING_LEVEL=INFO \
  linshuili-vllm-amd bash -lc '
    exec vllm serve /models/Qwen2.5-3B-Instruct \
      --served-model-name Qwen2.5-3B-Instruct \
      --dtype bfloat16 \
      --pipeline-parallel-size 2 \
      --tensor-parallel-size 1 \
      --distributed-executor-backend mp \
      --nnodes 2 \
      --node-rank 1 \
      --master-addr 127.0.0.1 \
      --master-port 29745 \
      --max-model-len 4096 \
      --max-num-seqs 8 \
      --max-num-batched-tokens 1024 \
      --gpu-memory-utilization 0.80 \
      --async-scheduling \
      --disable-custom-all-reduce \
      --generation-config vllm \
      --no-enable-prefix-caching \
      --headless
  '
```

N 卡侧（rank 0，对外服务端口）：

```bash
docker exec -it \
  -e PYTHONPATH=/work/patch_shm \
  -e GAN_SHM_SPSC=1 \
  -e GAN_SHM_SESSION=manual-lsl \
  -e GAN_SHM_LIBRARY=/work/native/libppshm.so \
  -e GAN_SHM_USE_FUTEX=1 \
  -e GAN_SHM_D2D_STAGE=1 \
  -e GAN_SHM_EARLY_META=0 \
  -e GAN_SHM_RAW_DMA=1 \
  -e GAN_SHM_PACKED_RECV=1 \
  -e GAN_SHM_NATIVE_PUBLISHER=1 \
  -e GAN_SHM_FUSED=1 \
  -e GAN_SHM_PROFILE=0 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_HOST_IP=127.0.0.1 \
  -e VLLM_CACHE_ROOT=/work/runs/manual-1008a/cache-1 \
  -e VLLM_PP_LAYER_PARTITION=18,18 \
  -e GLOO_SOCKET_IFNAME=lo \
  -e OMP_NUM_THREADS=4 \
  -e VLLM_LOGGING_LEVEL=INFO \
  linshuili-vllm-nv bash -lc '
    exec vllm serve /models/Qwen2.5-3B-Instruct \
      --served-model-name Qwen2.5-3B-Instruct \
      --dtype bfloat16 \
      --pipeline-parallel-size 2 \
      --tensor-parallel-size 1 \
      --distributed-executor-backend mp \
      --nnodes 2 \
      --node-rank 0 \
      --master-addr 127.0.0.1 \
      --master-port 29745 \
      --max-model-len 4096 \
      --max-num-seqs 8 \
      --max-num-batched-tokens 1024 \
      --gpu-memory-utilization 0.80 \
      --async-scheduling \
      --disable-custom-all-reduce \
      --generation-config vllm \
      --no-enable-prefix-caching \
      --host 127.0.0.1 \
      --port 18082
  '
```

## 创建 SHM 区域（改成你自己名字）

```bash
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, '/data/linshuili/gloo_infer/patch_shm')
from shm_native import create, unlink
Path('/data/linshuili/gloo_infer/patch_shm/manual-lsl').mkdir(parents=True, exist_ok=True)
create('manual-lsl', 'forward', 4, 65536, 8 * 1024 * 1024)
try:
    create('manual-lsl', 'feedback', 4, 65536, 1024 * 1024)
except BaseException:
    unlink('manual-lsl', 'forward')
    raise
print('SHM创建完成')
PY
```

## 结束之后记得删除 SHM

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, '/data/linshuili/gloo_infer/patch_shm')
from shm_native import unlink

for direction in ('forward', 'feedback'):
    try:
        unlink('manual-lsl', direction)
    except FileNotFoundError:
        pass

print('旧SHM已清理')
PY
```
