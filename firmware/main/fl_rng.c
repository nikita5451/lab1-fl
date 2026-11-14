#include "fl_rng.h"

uint32_t fl_rng_seed(uint32_t seed)
{
    uint32_t z = seed + 0x9E3779B9u;
    z ^= z >> 16;
    z *= 0x21F0AAADu;
    z ^= z >> 15;
    z *= 0x735A2D97u;
    z ^= z >> 15;
    return z != 0u ? z : 0x6D2B79F5u;
}

uint32_t fl_rng_next(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

float fl_rng_uniform(uint32_t *state)
{
    return (float)(fl_rng_next(state) >> 8) * (1.0f / 16777216.0f);
}

void fl_rng_shuffle_u16(uint16_t *a, uint32_t n, uint32_t *state)
{
    for (uint32_t i = n; i > 1; i--) {
        const uint32_t j = fl_rng_next(state) % i;
        const uint16_t t = a[i - 1];
        a[i - 1] = a[j];
        a[j] = t;
    }
}
