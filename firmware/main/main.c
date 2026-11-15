#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_mac.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_storage.h"
#include "fl_model.h"
#include "fl_proto.h"
#include "fl_store.h"

#define FW_VERSION "lr1-0.1"
#define LINE_MAX_LEN 1100
#define TASK_STACK 8192

static float g_params[FL_MAX_PARAMS], g_grads[FL_MAX_PARAMS], g_global[FL_MAX_PARAMS];
static fl_model_t g_model;
static fl_store_t g_store;
static fl_data_t g_data;
static fl_proto_t g_proto;
static char g_line[LINE_MAX_LEN];

static void port_write(const char *s, size_t n)
{
    while (n > 0) {
        int w = usb_serial_jtag_write_bytes(s, n > 64 ? 64 : n, pdMS_TO_TICKS(1000));
        if (w <= 0) return;                 
        s += w;
        n -= (size_t)w;
    }
}

static void write_line(void *ctx, const char *line)
{
    (void)ctx;
    port_write(line, strlen(line));
    port_write("\n", 1);
}

static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...)
{
    char b[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    write_line(NULL, b);
}

static uint64_t now_us(void *ctx) { (void)ctx; return (uint64_t)esp_timer_get_time(); }
static void reset_data(void *ctx) { (void)ctx; fl_store_split(&g_store, 12345u); }

static const char *chip_name(esp_chip_model_t m)
{
    switch (m) {
    case CHIP_ESP32S3: return "ESP32-S3";
    case CHIP_ESP32S2: return "ESP32-S2";
    case CHIP_ESP32C3: return "ESP32-C3";
    case CHIP_ESP32: return "ESP32";
    default: return "other";
    }
}

static void diag_info(void)
{
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    uint8_t mac[6];
    esp_efuse_mac_get_default(mac);
    say("OK info chip=%s rev=%d.%d cores=%d idf=%s fw=%s id=%02x%02x%02x%02x%02x%02x",
        chip_name(ci.model), ci.revision / 100, ci.revision % 100, ci.cores,
        esp_get_idf_version(), FW_VERSION, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

static void diag_heap(void)
{
    say("OK heap free=%u min_free=%u stack_free_min=%u stack_size=%u",
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
        (unsigned)uxTaskGetStackHighWaterMark(NULL), (unsigned)TASK_STACK);
}

static int handle_diag(const char *cmd)
{
    if (!strcmp(cmd, "help")) {
        say("OK help diag: help info heap restart | proto: INFO CRC INIT SETW GETW FIT EVAL STATS RESETSTATS RESET");
    } else if (!strcmp(cmd, "info")) {
        diag_info();
    } else if (!strcmp(cmd, "heap")) {
        diag_heap();
    } else if (!strcmp(cmd, "restart")) {
        say("OK restart");
        vTaskDelay(pdMS_TO_TICKS(100));
        esp_restart();
    } else {
        return 0;
    }
    return 1;
}

static void say_boot(void)
{
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    say("BOOT idf=%s chip=%s rev=%d.%d heap_free=%u fw=%s", esp_get_idf_version(),
        chip_name(ci.model), ci.revision / 100, ci.revision % 100,
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), FW_VERSION);
}

static void fl_task(void *arg)
{
    (void)arg;
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 2048;
    cfg.tx_buffer_size = 2048;
    usb_serial_jtag_driver_install(&cfg);

    say_boot();

    const uint8_t *blob = NULL;
    size_t len = 0;
    int rc = esp_storage_map(&blob, &len);
    if (rc == 0) rc = fl_store_open(&g_store, blob, len);   /* CRC при кожному старті */
    if (rc != 0) say("ERR storage rc=%d (чи прошито data_N.bin у fldata?)", rc);

    fl_model_init(&g_model, 64, 16, 10, g_params, g_grads);
    g_data = fl_store_as_data(&g_store);
    const fl_proto_io_t io = {write_line, now_us, reset_data, NULL};
    fl_proto_init(&g_proto, &io, &g_model, &g_data, g_global, rc == 0 ? g_store.n : 0,
                  FW_VERSION);
    say("OK ready n=%u", rc == 0 ? g_store.n : 0u);

    size_t pos = 0;
    int overflow = 0;
    int host_seen = 0;
    int64_t last_boot_ms = esp_timer_get_time() / 1000;
    uint8_t rx[128];
    for (;;) {
        int n = usb_serial_jtag_read_bytes(rx, sizeof rx, pdMS_TO_TICKS(50));
        if (n > 0) {
            host_seen = 1;
        } else if (!host_seen) {
            const int64_t now_ms = esp_timer_get_time() / 1000;
            if (now_ms - last_boot_ms >= 1500) {
                say_boot();
                last_boot_ms = now_ms;
            }
        }
        for (int i = 0; i < n; i++) {
            const char c = (char)rx[i];
            if (c == '\n') {
                g_line[pos] = '\0';
                while (pos > 0 && (g_line[pos - 1] == '\r' || g_line[pos - 1] == ' ')) {
                    g_line[--pos] = '\0';
                }
                if (overflow) {
                    say("ERR line_too_long");
                } else if (g_line[0] == '\0') {
                    /* порожній рядок ігноруємо */
                } else if (handle_diag(g_line)) {
                    /* help/info/heap/restart працюють завжди */
                } else if (rc != 0) {
                    say("ERR no_data");
                } else {
                    fl_proto_handle_line(&g_proto, g_line);
                }
                pos = 0;
                overflow = 0;
            } else if (pos < LINE_MAX_LEN - 1) {
                g_line[pos++] = c;
            } else {
                overflow = 1;
            }
        }
    }
}

void app_main(void)
{
    xTaskCreatePinnedToCore(fl_task, "fl", TASK_STACK, NULL, 5, NULL, 1);
}
