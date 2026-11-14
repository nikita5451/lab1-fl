#ifndef FL_PROTO_H
#define FL_PROTO_H
#include <stdint.h>
#include "fl_model.h"
#include "fl_train.h"

#define FL_WBYTES_MAX   (FL_MAX_PARAMS * 4)
#define FL_B64_LINE     1024                 

typedef struct {
    void (*write_line)(void *ctx, const char *line);   
    uint64_t (*now_us)(void *ctx);
    void (*reset_data)(void *ctx);                   
    void *ctx;
} fl_proto_io_t;

typedef struct {
    uint32_t fits, setw, getw, evals;     
    uint32_t w_rx_bytes, w_tx_bytes;     
} fl_stats_t;

typedef struct {
    fl_proto_io_t io;
    fl_model_t *model;
    fl_data_t *data;
    float *global;                       
    unsigned n_samples;
    const char *fw_version;
    fl_stats_t st;

    int rx_active;
    size_t rx_expect, rx_got;
    uint32_t rx_crc;
    uint8_t wbuf[FL_WBYTES_MAX];         
} fl_proto_t;

void fl_proto_init(fl_proto_t *p, const fl_proto_io_t *io, fl_model_t *m,
                   fl_data_t *d, float *global_buf, unsigned n_samples,
                   const char *fw_version);


void fl_proto_handle_line(fl_proto_t *p, char *line);

void fl_proto_init_weights(fl_proto_t *p, uint32_t seed);

uint32_t fl_proto_weights_crc(const fl_proto_t *p);
#endif
