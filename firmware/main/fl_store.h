#ifndef FL_STORE_H
#define FL_STORE_H
#include <stddef.h>
#include <stdint.h>
#include "fl_train.h"

#define FL_STORE_MAX_SAMPLES 1024  

typedef struct {
    const uint8_t *payload;   
    unsigned n, n_feat, n_cls;
    uint16_t train_idx[FL_STORE_MAX_SAMPLES];
    uint16_t val_idx[FL_STORE_MAX_SAMPLES];
    uint32_t n_train, n_val;
} fl_store_t;

int fl_store_open(fl_store_t *s, const uint8_t *blob, size_t len);

void fl_store_split(fl_store_t *s, uint32_t seed);

void fl_store_get_sample(void *ctx, uint32_t idx, float *x, int *y);

fl_data_t fl_store_as_data(fl_store_t *s);
#endif
