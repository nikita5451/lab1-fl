#include "fl_proto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fl_b64.h"
#include "fl_crc.h"
#include "fl_rng.h"

static void pack_le(const float *p, size_t n, uint8_t *out)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &p[i], 4);
        out[4 * i + 0] = (uint8_t)u;
        out[4 * i + 1] = (uint8_t)(u >> 8);
        out[4 * i + 2] = (uint8_t)(u >> 16);
        out[4 * i + 3] = (uint8_t)(u >> 24);
    }
}

static void unpack_le(const uint8_t *in, size_t n, float *p)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t u = (uint32_t)in[4 * i] | ((uint32_t)in[4 * i + 1] << 8) |
                     ((uint32_t)in[4 * i + 2] << 16) | ((uint32_t)in[4 * i + 3] << 24);
        memcpy(&p[i], &u, 4);
    }
}

static void out(fl_proto_t *p, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#include <stdarg.h>
static void out(fl_proto_t *p, const char *fmt, ...)
{
    char buf[FL_B64_LINE + 16];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    p->io.write_line(p->io.ctx, buf);
}

static uint64_t now(fl_proto_t *p) { return p->io.now_us(p->io.ctx); }

static void ev(fl_proto_t *p, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void ev(fl_proto_t *p, const char *fmt, ...)
{
    char body[200], line[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof body, fmt, ap);
    va_end(ap);
    snprintf(line, sizeof line, "EV t=%u %s", (unsigned)(now(p) / 1000u), body);
    p->io.write_line(p->io.ctx, line);
}

uint32_t fl_proto_weights_crc(const fl_proto_t *p)
{
    static uint8_t tmp[FL_WBYTES_MAX];
    pack_le(p->model->p, p->model->n_params, tmp);
    return fl_crc32(0, tmp, p->model->n_params * 4);
}

void fl_proto_init_weights(fl_proto_t *p, uint32_t seed)
{
    fl_model_init_weights(p->model, seed);
    memcpy(p->global, p->model->p, p->model->n_params * sizeof(float));
}

void fl_proto_init(fl_proto_t *p, const fl_proto_io_t *io, fl_model_t *m,
                   fl_data_t *d, float *global_buf, unsigned n_samples,
                   const char *fw_version)
{
    memset(p, 0, sizeof *p);
    p->io = *io;
    p->model = m;
    p->data = d;
    p->global = global_buf;
    p->n_samples = n_samples;
    p->fw_version = fw_version;
    fl_proto_init_weights(p, 1);
}

static unsigned lines_for(size_t bytes)
{
    const size_t per = FL_B64_LINE / 4 * 3;              
    return (unsigned)((bytes + per - 1) / per);
}

static void cmd_info(fl_proto_t *p)
{
    const fl_model_t *m = p->model;
    out(p, "OK INFO in=%d hidden=%d out=%d params=%zu bytes=%zu n=%u train=%u val=%u fw=%s",
        m->n_in, m->n_hidden, m->n_out, m->n_params, m->n_params * 4, p->n_samples,
        (unsigned)p->data->n_train, (unsigned)p->data->n_val, p->fw_version);
}

static void cmd_getw(fl_proto_t *p)
{
    const size_t bytes = p->model->n_params * 4;
    pack_le(p->model->p, p->model->n_params, p->wbuf);
    const uint32_t crc = fl_crc32(0, p->wbuf, bytes);
    const unsigned nl = lines_for(bytes);
    out(p, "OK GETW bytes=%zu crc=%08x lines=%u", bytes, (unsigned)crc, nl);
    const size_t per = FL_B64_LINE / 4 * 3;
    char b64[FL_B64_LINE + 8];
    for (size_t off = 0; off < bytes; off += per) {
        size_t n = bytes - off < per ? bytes - off : per;
        fl_b64_encode(p->wbuf + off, n, b64);
        out(p, "W %s", b64);
    }
    p->st.getw++;
    p->st.w_tx_bytes += (uint32_t)bytes;
    ev(p, "op=getw bytes=%zu crc=%08x", bytes, (unsigned)crc);
}

static void setw_finish(fl_proto_t *p)
{
    p->rx_active = 0;
    const uint32_t crc = fl_crc32(0, p->wbuf, p->rx_got);
    if (crc != p->rx_crc) {
        out(p, "ERR SETW crc got=%08x want=%08x", (unsigned)crc, (unsigned)p->rx_crc);
        return;
    }
    unpack_le(p->wbuf, p->model->n_params, p->model->p);
    memcpy(p->global, p->model->p, p->model->n_params * sizeof(float));
    p->st.setw++;
    p->st.w_rx_bytes += (uint32_t)p->rx_got;
    ev(p, "op=setw bytes=%zu crc=%08x", p->rx_got, (unsigned)crc);
    out(p, "OK SETW bytes=%zu crc=%08x", p->rx_got, (unsigned)crc);
}

static void cmd_setw(fl_proto_t *p, char *args)
{
    char *e1, *e2;
    const unsigned long bytes = strtoul(args, &e1, 10);
    const unsigned long crc = strtoul(e1, &e2, 16);
    if (e1 == args || e2 == e1 || bytes != p->model->n_params * 4) {
        out(p, "ERR SETW bad_args expected_bytes=%zu", p->model->n_params * 4);
        return;
    }
    p->rx_active = 1;
    p->rx_expect = bytes;
    p->rx_got = 0;
    p->rx_crc = (uint32_t)crc;
    out(p, "OK SETW ready lines=%u", lines_for(bytes));
}

static void rx_chunk(fl_proto_t *p, const char *b64)
{
    const int n = fl_b64_decode(b64, strlen(b64), p->wbuf + p->rx_got,
                                p->rx_expect - p->rx_got);
    if (n < 0) {
        p->rx_active = 0;
        out(p, "ERR SETW bad_chunk");
        return;
    }
    p->rx_got += (size_t)n;
    if (p->rx_got == p->rx_expect) setw_finish(p);
}

static void cmd_fit(fl_proto_t *p, char *args)
{
    char *e;
    const long epochs = strtol(args, &e, 10);
    if (e == args || epochs < 1 || epochs > 1000) {
        out(p, "ERR FIT usage: FIT <epochs> <lr> [batch] [seed] [clip]");
        return;
    }
    char *a = e;
    const float lr = strtof(a, &e);
    if (e == a || !(lr > 0.0f)) {
        out(p, "ERR FIT usage: FIT <epochs> <lr> [batch] [seed] [clip]");
        return;
    }
    a = e;
    long batch = strtol(a, &e, 10);
    if (e == a) batch = 16;
    a = e;
    unsigned long seed = strtoul(a, &e, 10);
    if (e == a) seed = 1;
    a = e;
    float clip = strtof(a, &e);
    if (e == a) clip = 0.0f;

    ev(p, "op=fit_begin epochs=%ld lr=%g batch=%ld n=%u", epochs, (double)lr, batch,
       (unsigned)p->data->n_train);
    const uint64_t t0 = now(p);
    const fl_result_t tr = fl_train(p->model, p->data, (int)epochs, lr, (int)batch,
                                    (uint32_t)seed);
    const uint64_t us = now(p) - t0;
    float scale = 1.0f;
    if (clip > 0.0f) scale = fl_clip_update(p->model, p->global, clip);
    const float dw = fl_model_delta_norm(p->model, p->global);
    const fl_result_t va = fl_eval(p->model, p->data, 1);   /* FL-04: звіт по val */
    p->st.fits++;
    ev(p, "op=fit_end fits=%u ms=%u loss=%.4f acc=%.4f dw=%.4f clip=%.4f",
       (unsigned)p->st.fits, (unsigned)(us / 1000u), (double)tr.loss, (double)tr.acc,
       (double)dw, (double)scale);
    out(p, "OK FIT n=%u loss=%.6f acc=%.6f val_loss=%.6f val_acc=%.6f ms=%u us=%u dw=%.6f",
        (unsigned)p->data->n_train, (double)tr.loss, (double)tr.acc, (double)va.loss,
        (double)va.acc, (unsigned)(us / 1000u), (unsigned)us, (double)dw);
}

static void cmd_eval(fl_proto_t *p)
{
    const uint64_t t0 = now(p);
    const fl_result_t r = fl_eval(p->model, p->data, 1);
    const uint64_t us = now(p) - t0;
    p->st.evals++;
    ev(p, "op=eval split=val n=%u acc=%.4f ms=%u", (unsigned)r.n, (double)r.acc,
       (unsigned)(us / 1000u));
    out(p, "OK EVAL split=val n=%u loss=%.6f acc=%.6f ms=%u us=%u", (unsigned)r.n,
        (double)r.loss, (double)r.acc, (unsigned)(us / 1000u), (unsigned)us);
}

void fl_proto_handle_line(fl_proto_t *p, char *line)
{
    while (*line == ' ' || *line == '\t') line++;
    size_t L = strlen(line);
    while (L && (line[L - 1] == '\r' || line[L - 1] == '\n' || line[L - 1] == ' ')) {
        line[--L] = '\0';
    }
    if (L == 0) return;                                   

    if (p->rx_active) {                                  
        if (line[0] == 'W' && line[1] == ' ') {
            rx_chunk(p, line + 2);
        } else {
            p->rx_active = 0;
            out(p, "ERR SETW aborted got=%zu of %zu", p->rx_got, p->rx_expect);
        }
        return;
    }

    char *args = line;
    while (*args && *args != ' ') args++;
    if (*args) *args++ = '\0';

    if (!strcmp(line, "INFO")) cmd_info(p);
    else if (!strcmp(line, "CRC")) out(p, "OK CRC crc=%08x", (unsigned)fl_proto_weights_crc(p));
    else if (!strcmp(line, "INIT")) {
        const unsigned long seed = strtoul(args, NULL, 10);
        fl_proto_init_weights(p, (uint32_t)seed);
        out(p, "OK INIT seed=%lu crc=%08x", seed, (unsigned)fl_proto_weights_crc(p));
    }
    else if (!strcmp(line, "SETW")) cmd_setw(p, args);
    else if (!strcmp(line, "GETW")) cmd_getw(p);
    else if (!strcmp(line, "FIT")) cmd_fit(p, args);
    else if (!strcmp(line, "EVAL")) cmd_eval(p);
    else if (!strcmp(line, "STATS")) {
        out(p, "OK STATS fits=%u setw=%u getw=%u evals=%u w_rx_bytes=%u w_tx_bytes=%u",
            (unsigned)p->st.fits, (unsigned)p->st.setw, (unsigned)p->st.getw,
            (unsigned)p->st.evals, (unsigned)p->st.w_rx_bytes, (unsigned)p->st.w_tx_bytes);
    }
    else if (!strcmp(line, "RESET")) {
        if (p->io.reset_data) p->io.reset_data(p->io.ctx);   
        fl_proto_init_weights(p, 1);
        memset(&p->st, 0, sizeof p->st);
        out(p, "OK RESET crc=%08x", (unsigned)fl_proto_weights_crc(p));
    }
    else if (!strcmp(line, "RESETSTATS")) {
        memset(&p->st, 0, sizeof p->st);
        out(p, "OK RESETSTATS");
    }
    else out(p, "ERR unknown_command %s", line);
}
