#include <stdio.h>
#include <string.h>
#include <time.h>

#include "fl_proto.h"
#include "fl_store.h"

static uint8_t blob[64 * 1024];
static fl_store_t store;
static float params[FL_MAX_PARAMS], grads[FL_MAX_PARAMS], global[FL_MAX_PARAMS];
static fl_model_t model;
static fl_proto_t proto;

static void wl(void *c, const char *l) { (void)c; fputs(l, stdout); fputc('\n', stdout); fflush(stdout); }
static uint64_t now_us(void *c)
{
    (void)c;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

static void reset_data(void *c) { (void)c; fl_store_split(&store, 12345u); }

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s data.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    size_t len = fread(blob, 1, sizeof blob, f);
    fclose(f);
    int rc = fl_store_open(&store, blob, len);
    if (rc) { fprintf(stderr, "store_open rc=%d\n", rc); return 2; }
    fl_data_t data = fl_store_as_data(&store);
    fl_model_init(&model, 64, 16, 10, params, grads);
    fl_proto_io_t io = {wl, now_us, reset_data, NULL};
    fl_proto_init(&proto, &io, &model, &data, global, store.n, "sim-1");
    static char line[2048];
    while (fgets(line, sizeof line, stdin)) fl_proto_handle_line(&proto, line);
    return 0;
}
