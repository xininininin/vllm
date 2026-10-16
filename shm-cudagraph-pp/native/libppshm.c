#define _GNU_SOURCE
#include "ppshm.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define PP_MAGIC 0x50505348u
#define PP_VERSION 4u
#define PP_MAX_SLOTS 8u
#define PP_PAGE 4096u

typedef struct {
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
    _Atomic uint32_t pushed;
    _Atomic uint32_t popped;
    uint32_t ids[PP_MAX_SLOTS];
} pp_queue;

typedef struct {
    uint64_t sequence;
    uint32_t metadata_bytes;
    uint32_t payload_bytes;
} pp_slot_header;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t slots;
    uint32_t metadata_capacity;
    uint32_t payload_capacity;
    uint32_t metadata_span;
    uint32_t payload_span;
    uint32_t slot_stride;
    uint64_t total_bytes;
    _Atomic int error;
    pp_queue free_queue;
    pp_queue metadata_queue;
    pp_queue ready_queue;
} pp_shared;

typedef struct {
    int fd;
    size_t length;
    pp_shared *shared;
} pp_region;

static uint32_t align_page(uint32_t value) {
    return (value + PP_PAGE - 1u) & ~(PP_PAGE - 1u);
}

static uint32_t control_span(void) {
    return align_page((uint32_t)sizeof(pp_shared));
}

static int valid_name(const char *name) {
    if (name == NULL || name[0] != '/' || name[1] == '\0') return 0;
    for (const char *p = name + 1; *p; ++p) {
        if (!( (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
               (*p >= '0' && *p <= '9') || *p == '_' || *p == '-')) return 0;
    }
    return 1;
}

int pp_create(const char *name, uint32_t slots, uint32_t metadata_bytes,
              uint32_t payload_bytes) {
    if (!valid_name(name) || slots < 2 || slots > PP_MAX_SLOTS ||
        metadata_bytes == 0 || payload_bytes == 0 ||
        metadata_bytes > UINT32_MAX - PP_PAGE ||
        payload_bytes > UINT32_MAX - PP_PAGE) return -EINVAL;
    uint32_t meta_span = align_page((uint32_t)sizeof(pp_slot_header) + metadata_bytes);
    uint32_t payload_span = align_page(payload_bytes);
    uint64_t stride = (uint64_t)meta_span + payload_span;
    uint64_t length = control_span() + slots * stride;
    if (length > SIZE_MAX || stride > UINT32_MAX) return -EOVERFLOW;

    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) return -errno;
    if (ftruncate(fd, (off_t)length) != 0) {
        int error = -errno;
        close(fd);
        shm_unlink(name);
        return error;
    }
    void *base = mmap(NULL, (size_t)length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        int error = -errno;
        close(fd);
        shm_unlink(name);
        return error;
    }
    pp_shared *shared = base;
    if (!atomic_is_lock_free(&shared->free_queue.head) ||
        !atomic_is_lock_free(&shared->free_queue.tail)) {
        munmap(base, (size_t)length);
        close(fd);
        shm_unlink(name);
        return -ENOTSUP;
    }
    shared->version = PP_VERSION;
    shared->slots = slots;
    shared->metadata_capacity = metadata_bytes;
    shared->payload_capacity = payload_bytes;
    shared->metadata_span = meta_span;
    shared->payload_span = payload_span;
    shared->slot_stride = (uint32_t)stride;
    shared->total_bytes = length;
    atomic_store_explicit(&shared->free_queue.head, 0, memory_order_relaxed);
    atomic_store_explicit(&shared->free_queue.tail, slots, memory_order_relaxed);
    for (uint32_t i = 0; i < slots; ++i) shared->free_queue.ids[i] = i;
    atomic_thread_fence(memory_order_release);
    shared->magic = PP_MAGIC; /* Publish initialized object last. */
    munmap(base, (size_t)length);
    close(fd);
    return 0;
}

void *pp_open(const char *name) {
    if (!valid_name(name)) { errno = EINVAL; return NULL; }
    int fd = shm_open(name, O_RDWR, 0);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)control_span()) {
        close(fd); errno = EINVAL; return NULL;
    }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { close(fd); return NULL; }
    pp_shared *shared = base;
    if (shared->magic != PP_MAGIC || shared->version != PP_VERSION ||
        shared->slots < 2 || shared->slots > PP_MAX_SLOTS ||
        shared->total_bytes != (uint64_t)st.st_size ||
        (uint64_t)shared->metadata_span < sizeof(pp_slot_header) + (uint64_t)shared->metadata_capacity ||
        shared->payload_span < shared->payload_capacity ||
        (uint64_t)shared->slot_stride != (uint64_t)shared->metadata_span + shared->payload_span ||
        shared->total_bytes != control_span() + (uint64_t)shared->slots * shared->slot_stride) {
        munmap(base, (size_t)st.st_size);
        close(fd); errno = EPROTO; return NULL;
    }
    pp_region *region = malloc(sizeof(*region));
    if (!region) { munmap(base, (size_t)st.st_size); close(fd); errno = ENOMEM; return NULL; }
    region->fd = fd;
    region->length = (size_t)st.st_size;
    region->shared = shared;
    return region;
}

int pp_close(void *pointer) {
    if (!pointer) return -EINVAL;
    pp_region *region = pointer;
    int result = munmap(region->shared, region->length) == 0 ? 0 : -errno;
    if (close(region->fd) != 0 && result == 0) result = -errno;
    free(region);
    return result;
}

int pp_unlink(const char *name) {
    if (!valid_name(name)) return -EINVAL;
    return shm_unlink(name) == 0 ? 0 : -errno;
}

static int futex_wait(_Atomic uint32_t *word, uint32_t expected,
                      const struct timespec *timeout) {
    return (int)syscall(SYS_futex, (uint32_t *)word, FUTEX_WAIT,
                        expected, timeout, NULL, 0);
}

static void futex_wake(_Atomic uint32_t *word) {
    syscall(SYS_futex, (uint32_t *)word, FUTEX_WAKE, 1, NULL, NULL, 0);
}

static uint64_t monotonic_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000u + (uint64_t)t.tv_nsec / 1000000u;
}

static int wait_change(_Atomic uint32_t *word, uint32_t seen,
                       uint64_t deadline, int use_futex) {
    uint64_t now = monotonic_ms();
    if (now >= deadline) return -ETIMEDOUT;
    uint64_t remaining = deadline - now;
    /* Bound each sleep so a peer's error state is observed promptly. */
    if (remaining > 100) remaining = 100;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = (long)remaining * 1000000L };
    if (use_futex) {
        if (futex_wait(word, seen, &ts) == 0 || errno == EAGAIN ||
            errno == EINTR || errno == ETIMEDOUT) return 0;
        return -errno;
    }
    ts.tv_nsec = 100000L; /* Ablation: old 100 us timed polling. */
    nanosleep(&ts, NULL);
    return 0;
}

static pp_queue *select_queue(pp_shared *shared, int kind) {
    if (kind == 0) return &shared->free_queue;
    if (kind == 1) return &shared->ready_queue;
    if (kind == 2) return &shared->metadata_queue;
    return NULL;
}

int pp_queue_pop(void *pointer, int ready, uint32_t *slot,
                 uint32_t timeout_ms, int use_futex) {
    if (!pointer || !slot || timeout_ms == 0) return -EINVAL;
    pp_shared *shared = ((pp_region *)pointer)->shared;
    pp_queue *q = select_queue(shared, ready);
    if (!q) return -EINVAL;
    uint64_t deadline = monotonic_ms() + timeout_ms;
    for (;;) {
        int error = atomic_load_explicit(&shared->error, memory_order_acquire);
        if (error) return -EPIPE;
        uint32_t head = atomic_load_explicit(&q->head, memory_order_relaxed);
        uint32_t tail = atomic_load_explicit(&q->tail, memory_order_acquire);
        if (head != tail) {
            uint32_t value = q->ids[head % shared->slots];
            if (value >= shared->slots) return -EPROTO;
            atomic_store_explicit(&q->head, head + 1, memory_order_release);
            atomic_fetch_add_explicit(&q->popped, 1, memory_order_release);
            futex_wake(&q->popped);
            *slot = value;
            return 0;
        }
        uint32_t seen = atomic_load_explicit(&q->pushed, memory_order_acquire);
        if (atomic_load_explicit(&q->tail, memory_order_acquire) != head) continue;
        int result = 0;
        if (atomic_load_explicit(&q->tail, memory_order_acquire) == head &&
            atomic_load_explicit(&q->pushed, memory_order_acquire) == seen)
            result = wait_change(&q->pushed, seen, deadline, use_futex);
        if (result) return result;
    }
}

int pp_queue_push(void *pointer, int ready, uint32_t slot,
                  uint32_t timeout_ms, int use_futex) {
    if (!pointer || timeout_ms == 0) return -EINVAL;
    pp_shared *shared = ((pp_region *)pointer)->shared;
    if (slot >= shared->slots) return -EINVAL;
    pp_queue *q = select_queue(shared, ready);
    if (!q) return -EINVAL;
    uint64_t deadline = monotonic_ms() + timeout_ms;
    for (;;) {
        int error = atomic_load_explicit(&shared->error, memory_order_acquire);
        if (error) return -EPIPE;
        uint32_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
        uint32_t head = atomic_load_explicit(&q->head, memory_order_acquire);
        if ((uint32_t)(tail - head) < shared->slots) {
            q->ids[tail % shared->slots] = slot;
            atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
            atomic_fetch_add_explicit(&q->pushed, 1, memory_order_release);
            futex_wake(&q->pushed);
            return 0;
        }
        uint32_t seen = atomic_load_explicit(&q->popped, memory_order_acquire);
        if ((uint32_t)(tail - atomic_load_explicit(&q->head, memory_order_acquire)) <
            shared->slots) continue;
        int result = 0;
        if ((uint32_t)(tail - atomic_load_explicit(&q->head, memory_order_acquire)) >=
                shared->slots &&
            atomic_load_explicit(&q->popped, memory_order_acquire) == seen)
            result = wait_change(&q->popped, seen, deadline, use_futex);
        if (result) return result;
    }
}

static pp_slot_header *slot_header(pp_region *region, uint32_t slot) {
    if (!region || slot >= region->shared->slots) return NULL;
    return (pp_slot_header *)((char *)region->shared + control_span() +
                              (size_t)slot * region->shared->slot_stride);
}

/* A GPU host callback must never wait on work that follows it on the stream.
 * Valid slot ownership guarantees room in the destination queue. If this
 * invariant is broken, report an error instead of blocking the callback. */
int pp_queue_try_push(void *pointer, int ready, uint32_t slot) {
    if (!pointer) return -EINVAL;
    pp_shared *shared = ((pp_region *)pointer)->shared;
    if (slot >= shared->slots) return -EINVAL;
    pp_queue *q = select_queue(shared, ready);
    if (!q) return -EINVAL;
    if (atomic_load_explicit(&shared->error, memory_order_acquire)) return -EPIPE;
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
    uint32_t head = atomic_load_explicit(&q->head, memory_order_acquire);
    if ((uint32_t)(tail - head) >= shared->slots) return -EAGAIN;
    q->ids[tail % shared->slots] = slot;
    atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
    atomic_fetch_add_explicit(&q->pushed, 1, memory_order_release);
    futex_wake(&q->pushed);
    return 0;
}

int pp_slot_write_header(void *pointer, uint32_t slot, uint64_t sequence,
                         uint32_t metadata_bytes, uint32_t payload_bytes) {
    pp_region *region = pointer;
    pp_slot_header *header = slot_header(region, slot);
    if (!header || !sequence || metadata_bytes > region->shared->metadata_capacity ||
        payload_bytes > region->shared->payload_capacity) return -EINVAL;
    header->sequence = sequence;
    header->metadata_bytes = metadata_bytes;
    header->payload_bytes = payload_bytes;
    return 0;
}

int pp_slot_read_header(void *pointer, uint32_t slot, uint64_t *sequence,
                        uint32_t *metadata_bytes, uint32_t *payload_bytes) {
    pp_slot_header *header = slot_header(pointer, slot);
    if (!header || !sequence || !metadata_bytes || !payload_bytes) return -EINVAL;
    pp_region *region = pointer;
    if (header->metadata_bytes > region->shared->metadata_capacity ||
        header->payload_bytes > region->shared->payload_capacity) return -EPROTO;
    *sequence = header->sequence;
    *metadata_bytes = header->metadata_bytes;
    *payload_bytes = header->payload_bytes;
    return 0;
}

void *pp_slot_metadata(void *pointer, uint32_t slot) {
    pp_slot_header *header = slot_header(pointer, slot);
    return header ? (char *)header + sizeof(*header) : NULL;
}

void *pp_slot_payload(void *pointer, uint32_t slot) {
    pp_region *region = pointer;
    pp_slot_header *header = slot_header(region, slot);
    return header ? (char *)header + region->shared->metadata_span : NULL;
}

uint32_t pp_slot_count(void *pointer) { return ((pp_region *)pointer)->shared->slots; }
uint32_t pp_metadata_capacity(void *pointer) { return ((pp_region *)pointer)->shared->metadata_capacity; }
uint32_t pp_payload_capacity(void *pointer) { return ((pp_region *)pointer)->shared->payload_capacity; }
uint32_t pp_payload_span(void *pointer) { return ((pp_region *)pointer)->shared->payload_span; }

int pp_set_error(void *pointer, int code) {
    if (!pointer || code == 0) return -EINVAL;
    pp_shared *shared = ((pp_region *)pointer)->shared;
    int expected = 0;
    /* Preserve the first failure; subsequent cleanup must not hide it. */
    atomic_compare_exchange_strong_explicit(&shared->error, &expected, code,
                                            memory_order_acq_rel,
                                            memory_order_acquire);
    pp_queue *queues[] = { &shared->free_queue, &shared->metadata_queue,
                           &shared->ready_queue };
    for (size_t i = 0; i < 3; ++i) {
        atomic_fetch_add_explicit(&queues[i]->pushed, 1, memory_order_release);
        atomic_fetch_add_explicit(&queues[i]->popped, 1, memory_order_release);
        futex_wake(&queues[i]->pushed);
        futex_wake(&queues[i]->popped);
    }
    return 0;
}

int pp_get_error(void *pointer) {
    if (!pointer) return -EINVAL;
    return atomic_load_explicit(&((pp_region *)pointer)->shared->error,
                                memory_order_acquire);
}
