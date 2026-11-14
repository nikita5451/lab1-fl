#include "fl_crc.h"

uint32_t fl_crc32(uint32_t crc, const void *data, size_t n)
{
    const uint8_t *p = data;
    crc = ~crc;
    while (n--) {
        crc ^= *p++;
        for (int k = 0; k < 8; k++) {
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
        }
    }
    return ~crc;
}
