#include "esp_storage.h"

#include "esp_partition.h"

int esp_storage_map(const uint8_t **blob, size_t *len)
{
    const esp_partition_t *part = esp_partition_find_first(
        (esp_partition_type_t)0x40, ESP_PARTITION_SUBTYPE_ANY, "fldata");
    if (part == NULL) return -1;
    const void *ptr = NULL;
    esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) != ESP_OK) {
        return -2;
    }
    *blob = ptr;
    *len = part->size;
    return 0;
}
