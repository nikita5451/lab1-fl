#ifndef FL_RNG_H
#define FL_RNG_H
#include <stdint.h>

uint32_t fl_rng_seed(uint32_t seed);

uint32_t fl_rng_next(uint32_t *state);

float fl_rng_uniform(uint32_t *state);

void fl_rng_shuffle_u16(uint16_t *a, uint32_t n, uint32_t *state);

#endif 
