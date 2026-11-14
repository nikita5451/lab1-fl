#ifndef FL_B64_H
#define FL_B64_H
#include <stddef.h>
#include <stdint.h>

size_t fl_b64_encode(const uint8_t *in, size_t n, char *out);
int fl_b64_decode(const char *in, size_t n, uint8_t *out, size_t cap);
#endif
