#ifndef PPSHM_H
#define PPSHM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All functions return zero on success or a negative errno value. */
int pp_create(const char *name, uint32_t slots, uint32_t metadata_bytes,
              uint32_t payload_bytes);
void *pp_open(const char *name);
int pp_close(void *region);
int pp_unlink(const char *name);
/* Queue kind: 0=free, 1=DMA-complete ready, 2=metadata published. */
int pp_queue_pop(void *region, int kind, uint32_t *slot,
                 uint32_t timeout_ms, int use_futex);
int pp_queue_push(void *region, int kind, uint32_t slot,
                  uint32_t timeout_ms, int use_futex);
int pp_slot_write_header(void *region, uint32_t slot, uint64_t sequence,
                         uint32_t metadata_bytes, uint32_t payload_bytes);
int pp_slot_read_header(void *region, uint32_t slot, uint64_t *sequence,
                        uint32_t *metadata_bytes, uint32_t *payload_bytes);
void *pp_slot_metadata(void *region, uint32_t slot);
void *pp_slot_payload(void *region, uint32_t slot);
uint32_t pp_slot_count(void *region);
uint32_t pp_metadata_capacity(void *region);
uint32_t pp_payload_capacity(void *region);
uint32_t pp_payload_span(void *region);
int pp_set_error(void *region, int code);
int pp_get_error(void *region);

#ifdef __cplusplus
}
#endif
#endif
