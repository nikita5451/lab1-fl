#include "fl_store.h"

#include "fl_crc.h"
#include "fl_model.h"
#include "fl_rng.h"

static unsigned rd16(const uint8_t *p) { return p[0] | ((unsigned)p[1] << 8); }
static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

int fl_store_open(fl_store_t *s, const uint8_t *blob, size_t len)
{
    if (len < 20 || rd32(blob) != 0x53444C46u || rd16(blob + 4) != 1) return -1;
    s->n_feat = rd16(blob + 6);
    s->n_cls = rd16(blob + 8);
    s->n = rd32(blob + 12);
    const uint32_t crc = rd32(blob + 16);
    if (s->n_feat > FL_MAX_IN || s->n > FL_STORE_MAX_SAMPLES) return -3;
    const size_t bytes = (size_t)s->n * (s->n_feat + 1);
    if (len < 20 + bytes) return -1;
    if (fl_crc32(0, blob + 20, bytes) != crc) return -2;
    s->payload = blob + 20;
    fl_store_split(s, 12345u);
    return 0;
}

void fl_store_split(fl_store_t *s, uint32_t seed)
{
    for (unsigned i = 0; i < s->n; i++) s->train_idx[i] = (uint16_t)i;
    uint32_t st = fl_rng_seed(seed);
    fl_rng_shuffle_u16(s->train_idx, s->n, &st);        
    s->n_val = s->n / 5;
    s->n_train = s->n - s->n_val;
    for (uint32_t i = 0; i < s->n_val; i++) {
        s->val_idx[i] = s->train_idx[s->n_train + i];
    }
}

void fl_store_get_sample(void *ctx, uint32_t idx, float *x, int *y)
{
    const fl_store_t *s = ctx;
    const uint8_t *r = s->payload + (size_t)idx * (s->n_feat + 1);
    for (unsigned i = 0; i < s->n_feat; i++) x[i] = (float)r[i] * (1.0f / 255.0f);
    *y = r[s->n_feat];
}

fl_data_t fl_store_as_data(fl_store_t *s)
{
    fl_data_t d = {
        .get_sample = fl_store_get_sample, .ctx = s,
        .train_idx = s->train_idx, .n_train = s->n_train,
        .val_idx = s->val_idx, .n_val = s->n_val,
    };
    return d;
}
