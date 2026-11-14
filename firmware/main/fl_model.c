#include "fl_model.h"

#include <math.h>
#include <string.h>

#include "fl_rng.h"

size_t fl_model_n_params(int n_in, int n_hidden, int n_out)
{
    return (size_t)n_hidden * n_in + n_hidden
         + (size_t)n_out * n_hidden + n_out;
}

int fl_model_init(fl_model_t *m, int n_in, int n_hidden, int n_out,
                  float *params, float *grad)
{
    if (n_in < 1 || n_in > FL_MAX_IN || n_hidden < 1 ||
        n_hidden > FL_MAX_HIDDEN || n_out < 2 || n_out > FL_MAX_OUT ||
        params == NULL || grad == NULL) {
        return -1;
    }
    memset(m, 0, sizeof(*m));
    m->n_in = n_in;
    m->n_hidden = n_hidden;
    m->n_out = n_out;
    m->n_params = fl_model_n_params(n_in, n_hidden, n_out);
    m->p = params;
    m->g = grad;

    float *p = params, *g = grad;
    m->w1 = p;  m->gw1 = g;
    p += (size_t)n_hidden * n_in;
    g += (size_t)n_hidden * n_in;
    m->b1 = p;  m->gb1 = g;
    p += n_hidden;
    g += n_hidden;
    m->w2 = p;  m->gw2 = g;
    p += (size_t)n_out * n_hidden;
    g += (size_t)n_out * n_hidden;
    m->b2 = p;  m->gb2 = g;
    return 0;
}

void fl_model_init_weights(fl_model_t *m, uint32_t seed)
{
    uint32_t s = fl_rng_seed(seed);
    memset(m->p, 0, m->n_params * sizeof(float));   
    
    float lim = sqrtf(6.0f / (float)(m->n_in + m->n_hidden));
    for (size_t i = 0; i < (size_t)m->n_hidden * m->n_in; i++) {
        m->w1[i] = (2.0f * fl_rng_uniform(&s) - 1.0f) * lim;
    }
    lim = sqrtf(6.0f / (float)(m->n_hidden + m->n_out));
    for (size_t i = 0; i < (size_t)m->n_out * m->n_hidden; i++) {
        m->w2[i] = (2.0f * fl_rng_uniform(&s) - 1.0f) * lim;
    }
}

int fl_model_forward(fl_model_t *m, const float *x)
{
    for (int j = 0; j < m->n_hidden; j++) {
        const float *w = m->w1 + (size_t)j * m->n_in;
        float acc = m->b1[j];
        for (int i = 0; i < m->n_in; i++) {
            acc += w[i] * x[i];
        }
        m->hpre[j] = acc;
        m->h[j] = acc > 0.0f ? acc : 0.0f;
    }

    float logit[FL_MAX_OUT];
    float max = -INFINITY;
    for (int k = 0; k < m->n_out; k++) {
        const float *w = m->w2 + (size_t)k * m->n_hidden;
        float acc = m->b2[k];
        for (int j = 0; j < m->n_hidden; j++) {
            acc += w[j] * m->h[j];
        }
        logit[k] = acc;
        if (acc > max) {
            max = acc;
        }
    }

    float sum = 0.0f;
    for (int k = 0; k < m->n_out; k++) {
        m->prob[k] = expf(logit[k] - max);
        sum += m->prob[k];
    }
    int best = 0;
    for (int k = 0; k < m->n_out; k++) {
        m->prob[k] /= sum;
        if (m->prob[k] > m->prob[best]) {
            best = k;
        }
    }
    return best;
}

float fl_model_loss(const fl_model_t *m, int y)
{
    if (y < 0 || y >= m->n_out) {
        return 0.0f;
    }
    const float p = m->prob[y];
    return -logf(p > 1e-12f ? p : 1e-12f);
}

void fl_model_zero_grad(fl_model_t *m)
{
    memset(m->g, 0, m->n_params * sizeof(float));
}

float fl_model_accumulate(fl_model_t *m, const float *x, int y, int *ok)
{
    const int pred = fl_model_forward(m, x);
    if (ok != NULL) {
        *ok = (pred == y) ? 1 : 0;
    }
    const float loss = fl_model_loss(m, y);


    float dlogit[FL_MAX_OUT];
    for (int k = 0; k < m->n_out; k++) {
        dlogit[k] = m->prob[k] - (k == y ? 1.0f : 0.0f);
    }

    for (int k = 0; k < m->n_out; k++) {
        float *gw = m->gw2 + (size_t)k * m->n_hidden;
        const float d = dlogit[k];
        for (int j = 0; j < m->n_hidden; j++) {
            gw[j] += d * m->h[j];
        }
        m->gb2[k] += d;
    }

    for (int j = 0; j < m->n_hidden; j++) {
        if (m->hpre[j] <= 0.0f) {
            continue;                      
        }
        float d = 0.0f;
        for (int k = 0; k < m->n_out; k++) {
            d += m->w2[(size_t)k * m->n_hidden + j] * dlogit[k];
        }
        float *gw = m->gw1 + (size_t)j * m->n_in;
        for (int i = 0; i < m->n_in; i++) {
            gw[i] += d * x[i];
        }
        m->gb1[j] += d;
    }
    return loss;
}

void fl_model_sgd_step(fl_model_t *m, float lr, int batch)
{

    const float scale = lr / (float)(batch > 0 ? batch : 1);
    for (size_t i = 0; i < m->n_params; i++) {
        m->p[i] -= scale * m->g[i];
    }
}

float fl_model_delta_norm(const fl_model_t *m, const float *other)
{
    float sq = 0.0f;
    for (size_t i = 0; i < m->n_params; i++) {
        const float d = m->p[i] - other[i];
        sq += d * d;
    }
    return sqrtf(sq);
}
