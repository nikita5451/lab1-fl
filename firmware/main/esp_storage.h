#ifndef ESP_STORAGE_H
#define ESP_STORAGE_H
#include <stddef.h>
#include <stdint.h>

int esp_storage_map(const uint8_t **blob, size_t *len);
#endif
