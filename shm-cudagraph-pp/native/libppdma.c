#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SLOTS 8u
typedef int (*copy_fn)(void *, const void *, size_t, int, void *);
typedef int (*create_fn)(void **, unsigned int);
typedef int (*record_fn)(void *, void *);
typedef int (*event_fn)(void *);
typedef int (*wait_fn)(void *, void *, unsigned int);
typedef const char *(*name_fn)(int);
typedef int (*device_fn)(int);
typedef int (*queue_fn)(void *, int, uint32_t);
typedef int (*error_fn)(void *, int);
typedef struct { uint64_t destination, source, bytes; int32_t kind, padding; } gpu_copy;
typedef struct {
    void *event, *source_event, *region;
    uint64_t generation;
    int ready, use_event, busy;
} slot_job;
typedef struct {
    void *library, *control;
    copy_fn copy;
    create_fn create;
    record_fn record;
    event_fn sync, destroy, query;
    wait_fn wait;
    name_fn error_name;
    device_fn device;
    queue_fn push;
    error_fn fail;
    pthread_mutex_t mutex;
    pthread_cond_t available, drained;
    pthread_t thread;
    slot_job jobs[MAX_SLOTS];
    uint64_t source_generation[MAX_SLOTS];
    uint32_t ids[MAX_SLOTS], slots, head, tail, active;
    char direction[16];
    int device_id, stop, initialized, error;
    int trace;
} gpu_runtime;

static uint64_t monotonic_us(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)now.tv_nsec / 1000u;
}

static const char *queue_name(int ready) {
    return ready ? "ready" : "free";
}

static void trace_event(gpu_runtime *runtime, const char *phase,
                        uint32_t slot, int ready, uint64_t generation,
                        uint64_t elapsed_us, int result) {
    if (!runtime->trace) return;
    fprintf(stderr,
            "CROSS_PP_NATIVE direction=%s phase=%s slot=%u queue=%s generation=%llu elapsed_us=%llu result=%d\n",
            runtime->direction, phase, slot, queue_name(ready),
            (unsigned long long)generation, (unsigned long long)elapsed_us, result);
}

static void *publisher(void *data) {
    gpu_runtime *runtime = data;
    int result = runtime->device(runtime->device_id);
    pthread_mutex_lock(&runtime->mutex);
    runtime->error = result;
    runtime->initialized = 1;
    pthread_cond_broadcast(&runtime->drained);
    if (result) { pthread_mutex_unlock(&runtime->mutex); return NULL; }
    for (;;) {
        while (runtime->head == runtime->tail && !runtime->stop)
            pthread_cond_wait(&runtime->available, &runtime->mutex);
        if (runtime->head == runtime->tail && runtime->stop) break;
        uint32_t slot = runtime->ids[runtime->head++ % runtime->slots];
        slot_job job = runtime->jobs[slot];
        pthread_mutex_unlock(&runtime->mutex);
        uint64_t started = monotonic_us();
        trace_event(runtime, "EVENT_SYNC_BEGIN", slot, job.ready,
                    job.generation, 0, 0);
        result = job.use_event ? runtime->sync(job.event) : 0;
        trace_event(runtime, "EVENT_SYNC_END", slot, job.ready, job.generation,
                    monotonic_us() - started, result);
        pthread_mutex_lock(&runtime->mutex);
        /* Event completion precedes queue publication. Slot reuse cannot
         * race its event because the peer must first return ownership. */
        runtime->jobs[slot].busy = 0;
        runtime->active--;
        if (!result) result = runtime->push(job.region, job.ready, slot);
        if (result) { runtime->error = result; runtime->fail(job.region, EIO); }
        pthread_cond_broadcast(&runtime->drained);
    }
    pthread_mutex_unlock(&runtime->mutex);
    return NULL;
}

void *pp_dma_open(const char *path, int hip, const char *control_path,
                  uint32_t slots, int device_id, const char *direction) {
    if (slots < 2 || slots > MAX_SLOTS) return NULL;
    gpu_runtime *runtime = calloc(1, sizeof(*runtime));
    if (!runtime) return NULL;
    runtime->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    runtime->control = dlopen(control_path, RTLD_NOW | RTLD_LOCAL);
    if (!runtime->library || !runtime->control) goto failed;
    runtime->copy = (copy_fn)dlsym(runtime->library, hip ? "hipMemcpyAsync" : "cudaMemcpyAsync");
    runtime->create = (create_fn)dlsym(runtime->library, hip ? "hipEventCreateWithFlags" : "cudaEventCreateWithFlags");
    runtime->record = (record_fn)dlsym(runtime->library, hip ? "hipEventRecord" : "cudaEventRecord");
    runtime->sync = (event_fn)dlsym(runtime->library, hip ? "hipEventSynchronize" : "cudaEventSynchronize");
    runtime->destroy = (event_fn)dlsym(runtime->library, hip ? "hipEventDestroy" : "cudaEventDestroy");
    runtime->query = (event_fn)dlsym(runtime->library, hip ? "hipEventQuery" : "cudaEventQuery");
    runtime->wait = (wait_fn)dlsym(runtime->library, hip ? "hipStreamWaitEvent" : "cudaStreamWaitEvent");
    runtime->error_name = (name_fn)dlsym(runtime->library, hip ? "hipGetErrorName" : "cudaGetErrorName");
    runtime->device = (device_fn)dlsym(runtime->library, hip ? "hipSetDevice" : "cudaSetDevice");
    runtime->push = (queue_fn)dlsym(runtime->control, "pp_queue_try_push");
    runtime->fail = (error_fn)dlsym(runtime->control, "pp_set_error");
    if (!runtime->copy || !runtime->create || !runtime->record || !runtime->sync ||
        !runtime->destroy || !runtime->query || !runtime->wait || !runtime->error_name ||
        !runtime->device || !runtime->push || !runtime->fail) goto failed;
    runtime->slots = slots;
    runtime->device_id = device_id;
    snprintf(runtime->direction, sizeof(runtime->direction), "%s",
             direction && direction[0] ? direction : "unknown");
    runtime->trace = getenv("GAN_SHM_NATIVE_TRACE") != NULL &&
                     strcmp(getenv("GAN_SHM_NATIVE_TRACE"), "0") != 0;
    if (runtime->device(device_id)) goto failed;
    for (uint32_t i = 0; i < slots; ++i) {
        if (runtime->create(&runtime->jobs[i].event, 2u) ||
            runtime->create(&runtime->jobs[i].source_event, 2u)) goto failed;
    }
    pthread_mutex_init(&runtime->mutex, NULL);
    pthread_cond_init(&runtime->available, NULL);
    pthread_condattr_t cond_attr;
    pthread_condattr_init(&cond_attr);
    pthread_condattr_setclock(&cond_attr, CLOCK_MONOTONIC);
    pthread_cond_init(&runtime->drained, &cond_attr);
    pthread_condattr_destroy(&cond_attr);
    if (pthread_create(&runtime->thread, NULL, publisher, runtime)) {
        pthread_mutex_destroy(&runtime->mutex);
        pthread_cond_destroy(&runtime->available);
        pthread_cond_destroy(&runtime->drained);
        goto failed;
    }
    pthread_mutex_lock(&runtime->mutex);
    while (!runtime->initialized) pthread_cond_wait(&runtime->drained, &runtime->mutex);
    int error = runtime->error;
    pthread_mutex_unlock(&runtime->mutex);
    if (error) {
        pthread_join(runtime->thread, NULL);
        pthread_mutex_destroy(&runtime->mutex);
        pthread_cond_destroy(&runtime->available);
        pthread_cond_destroy(&runtime->drained);
        goto failed;
    }
    return runtime;
failed:
    if (runtime->destroy)
        for (uint32_t i = 0; i < slots; ++i)
            {
                if (runtime->jobs[i].event) runtime->destroy(runtime->jobs[i].event);
                if (runtime->jobs[i].source_event) runtime->destroy(runtime->jobs[i].source_event);
            }
    if (runtime->control) dlclose(runtime->control);
    if (runtime->library) dlclose(runtime->library);
    free(runtime);
    return NULL;
}

int pp_dma_batch(void *pointer, const gpu_copy *copies, uint32_t count, uint64_t stream) {
    if (!pointer || (!copies && count)) return -EINVAL;
    gpu_runtime *runtime = pointer;
    for (uint32_t i = 0; i < count; ++i) {
        if (!copies[i].bytes) continue;
        if (copies[i].kind < 1 || copies[i].kind > 3) return -EINVAL;
        int result = runtime->copy((void *)(uintptr_t)copies[i].destination,
            (const void *)(uintptr_t)copies[i].source, (size_t)copies[i].bytes,
            copies[i].kind, (void *)(uintptr_t)stream);
        if (result) return result;
    }
    return 0;
}

static int enqueue_locked(gpu_runtime *runtime, void *region, uint32_t slot,
                   int ready, uint64_t generation, uint64_t stream, int use_event) {
    if (!runtime || !region || slot >= runtime->slots) return -EINVAL;
    int result = 0;
    if (runtime->error || runtime->stop) result = -EPIPE;
    else if (runtime->jobs[slot].busy || runtime->tail - runtime->head >= runtime->slots)
        result = -EBUSY;
    else {
        if (use_event) result = runtime->record(runtime->jobs[slot].event,
                                                (void *)(uintptr_t)stream);
        if (!result) {
            runtime->jobs[slot].region = region;
            runtime->jobs[slot].ready = ready;
            runtime->jobs[slot].generation = generation;
            runtime->jobs[slot].use_event = use_event;
            runtime->jobs[slot].busy = 1;
            runtime->active++;
            runtime->ids[runtime->tail++ % runtime->slots] = slot;
            trace_event(runtime, "EVENT_ENQUEUED", slot, ready, generation, 0, 0);
            pthread_cond_signal(&runtime->available);
        }
    }
    return result;
}

static int enqueue(gpu_runtime *runtime, void *region, uint32_t slot,
                   int ready, uint64_t generation, uint64_t stream, int use_event) {
    if (!runtime) return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    int result = enqueue_locked(runtime, region, slot, ready, generation,
                                stream, use_event);
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

/* Caller already owns a free SHM slot. Its preceding generation therefore
 * completed both D2H and peer H2D; source_event may now be recorded again. */
int pp_dma_fused(void *pointer, void *region, uint32_t slot, uint64_t generation,
                 const gpu_copy *stage, uint32_t stage_count,
                 const gpu_copy *host, uint32_t host_count,
                 uint64_t model_stream, uint64_t copy_stream) {
    gpu_runtime *runtime = pointer;
    if (!runtime || !region || slot >= runtime->slots || !generation) return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    int result = 0;
    if (runtime->error || runtime->stop) result = -EPIPE;
    else if (runtime->jobs[slot].busy) result = -EBUSY;
    else if (generation <= runtime->source_generation[slot]) result = -EPROTO;
    else {
        trace_event(runtime, "FUSED_SUBMIT_BEGIN", slot, 1, generation, 0, 0);
        runtime->source_generation[slot] = generation;
        result = pp_dma_batch(pointer, stage, stage_count, model_stream);
        if (!result) result = runtime->record(runtime->jobs[slot].source_event,
                                              (void *)(uintptr_t)model_stream);
        if (!result) result = runtime->wait((void *)(uintptr_t)copy_stream,
                                            runtime->jobs[slot].source_event, 0);
        if (!result) result = pp_dma_batch(pointer, host, host_count, copy_stream);
        if (!result) result = enqueue_locked(runtime, region, slot, 1, generation,
                                             copy_stream, 1);
        trace_event(runtime, "FUSED_SUBMIT_END", slot, 1, generation, 0, result);
    }
    if (result) { runtime->error = result; runtime->fail(region, EIO); }
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

int pp_dma_source_wait(void *pointer, uint32_t slot, uint64_t generation,
                       uint64_t stream) {
    gpu_runtime *runtime = pointer;
    if (!runtime || slot >= runtime->slots) return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    int result = 0;
    if (generation > runtime->source_generation[slot]) result = -EPROTO;
    else if (generation == runtime->source_generation[slot])
        result = runtime->wait((void *)(uintptr_t)stream,
                               runtime->jobs[slot].source_event, 0);
    /* An older handle is already source-safe: reuse required a free slot. */
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

int pp_dma_source_query(void *pointer, uint32_t slot, uint64_t generation) {
    gpu_runtime *runtime = pointer;
    if (!runtime || slot >= runtime->slots) return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    int result = 1;
    if (generation > runtime->source_generation[slot]) result = -EPROTO;
    else if (generation == runtime->source_generation[slot]) {
        int status = runtime->query(runtime->jobs[slot].source_event);
        if (status) {
            const char *name = runtime->error_name(status);
            result = name && (strcmp(name, "cudaErrorNotReady") == 0 ||
                              strcmp(name, "hipErrorNotReady") == 0) ? 0 : -status;
        }
    }
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

int pp_dma_publish(void *pointer, void *region, uint32_t slot, int ready, uint64_t stream) {
    return enqueue(pointer, region, slot, ready, 0, stream, 1);
}
int pp_dma_release_cpu(void *pointer, void *region, uint32_t slot) {
    return enqueue(pointer, region, slot, 0, 0, 0, 0);
}
int pp_dma_drain(void *pointer, uint32_t timeout_ms) {
    if (!pointer) return -EINVAL;
    gpu_runtime *runtime = pointer;
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += timeout_ms / 1000;
    deadline.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) { deadline.tv_sec++; deadline.tv_nsec -= 1000000000L; }
    pthread_mutex_lock(&runtime->mutex);
    int result = 0;
    while (runtime->active && !result)
        result = pthread_cond_timedwait(&runtime->drained, &runtime->mutex, &deadline);
    if (!result && runtime->error) result = EIO;
    pthread_mutex_unlock(&runtime->mutex);
    return -result;
}
int pp_dma_close(void *pointer) {
    if (!pointer) return 0;
    gpu_runtime *runtime = pointer;
    int result = pp_dma_drain(pointer, 60000);
    if (result) return result;
    pthread_mutex_lock(&runtime->mutex);
    runtime->stop = 1;
    pthread_cond_signal(&runtime->available);
    pthread_mutex_unlock(&runtime->mutex);
    pthread_join(runtime->thread, NULL);
    for (uint32_t i = 0; i < runtime->slots; ++i) {
        runtime->destroy(runtime->jobs[i].event);
        runtime->destroy(runtime->jobs[i].source_event);
    }
    pthread_mutex_destroy(&runtime->mutex);
    pthread_cond_destroy(&runtime->available);
    pthread_cond_destroy(&runtime->drained);
    dlclose(runtime->library);
    dlclose(runtime->control);
    free(runtime);
    return 0;
}
