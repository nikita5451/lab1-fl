#include "fl_b64.h"

static const char T[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t fl_b64_encode(const uint8_t *in, size_t n, char *out)
{
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i + 1 < n) v |= (uint32_t)in[i + 1] << 8;
        if (i + 2 < n) v |= in[i + 2];
        out[o++] = T[(v >> 18) & 63];
        out[o++] = T[(v >> 12) & 63];
        out[o++] = (i + 1 < n) ? T[(v >> 6) & 63] : '=';
        out[o++] = (i + 2 < n) ? T[v & 63] : '=';
    }
    out[o] = '\0';
    return o;
}

static int val(char c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

int fl_b64_decode(const char *in, size_t n, uint8_t *out, size_t cap)
{
    if (n % 4 != 0) return -1;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 4) {
        int a = val(in[i]), b = val(in[i + 1]);
        int pc = (in[i + 2] == '=') + (in[i + 3] == '=');
        int c = in[i + 2] == '=' ? 0 : val(in[i + 2]);
        int d = in[i + 3] == '=' ? 0 : val(in[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0) return -1;
        if (pc && i + 4 != n) return -1;
        uint32_t v = ((uint32_t)a << 18) | ((uint32_t)b << 12) |
                     ((uint32_t)c << 6) | (uint32_t)d;
        if (o + (size_t)(3 - pc) > cap) return -1;
        out[o++] = (uint8_t)(v >> 16);
        if (pc < 2) out[o++] = (uint8_t)(v >> 8);
        if (pc < 1) out[o++] = (uint8_t)v;
    }
    return (int)o;
}
