#ifndef FL_CRC_H
#define FL_CRC_H
#include <stddef.h>
#include <stdint.h>

uint32_t fl_crc32(uint32_t crc, const void *data, size_t n);
#endif
