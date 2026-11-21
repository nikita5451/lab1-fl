#include "fl_train.h"

#include "fl_rng.h"

fl_result_t fl_train(fl_model_t *m, fl_data_t *d, int epochs, float lr,
                     int batch, uint32_t seed)
{
    fl_result_t r = {0, 0, 0.0f, 0.0f};
    float x[FL_MAX_IN];

    if (epochs < 1) {
        epochs = 1;
    }
    if (batch < 1) {
        batch = 1;
    }
    if (d->n_train == 0) {
        return r;
    }

    for (int e = 0; e < epochs; e++) {
        /* Перемішування щоепохи, а не один раз: інакше батчі завжди
         * однакові й SGD бачить ту саму послідовність градієнтів. */
        uint32_t s = fl_rng_seed(seed + (uint32_t)e);
        fl_rng_shuffle_u16(d->train_idx, d->n_train, &s);

        double loss_sum = 0.0;
        uint32_t correct = 0, seen = 0, steps = 0;

        for (uint32_t off = 0; off < d->n_train; off += (uint32_t)batch) {
            uint32_t b = d->n_train - off;
            if (b > (uint32_t)batch) {
                b = (uint32_t)batch;
            }
            fl_model_zero_grad(m);
            for (uint32_t i = 0; i < b; i++) {
                int y = 0, ok = 0;
                d->get_sample(d->ctx, d->train_idx[off + i], x, &y);
                loss_sum += fl_model_accumulate(m, x, y, &ok);
                correct += (uint32_t)ok;
                seen++;
            }
            fl_model_sgd_step(m, lr, (int)b);
            steps++;
        }
        
        /* Звітуємо останню епоху: вона відображає стан, у якому ваги
         * поїдуть на сервер. */
        r.n = seen;
        r.steps = steps;
        r.loss = seen ? (float)(loss_sum / seen) : 0.0f;
        r.acc = seen ? (float)correct / (float)seen : 0.0f;
    }
    return r;
}

fl_result_t fl_eval(fl_model_t *m, const fl_data_t *d, int on_val)
{
    fl_result_t r = {0, 0, 0.0f, 0.0f};
    float x[FL_MAX_IN];

    const uint16_t *idx = on_val ? d->val_idx : d->train_idx;
    const uint32_t n = on_val ? d->n_val : d->n_train;
    if (idx == NULL || n == 0) {
        return r;
    }

    double loss_sum = 0.0;
    uint32_t correct = 0;
    for (uint32_t i = 0; i < n; i++) {
        int y = 0;
        d->get_sample(d->ctx, idx[i], x, &y);
        const int pred = fl_model_forward(m, x);
        loss_sum += fl_model_loss(m, y);
        if (pred == y) {
            correct++;
        }
    }
    r.n = n;
    r.loss = (float)(loss_sum / n);
    r.acc = (float)correct / (float)n;
    return r;
}

float fl_clip_update(fl_model_t *m, const float *global, float max_norm)
{
    if (max_norm <= 0.0f || global == NULL) {
        return 1.0f;
    }
    const float norm = fl_model_delta_norm(m, global);
    if (norm <= max_norm || norm == 0.0f) {
        return 1.0f;
    }
    const float scale = max_norm / norm;
    for (size_t i = 0; i < m->n_params; i++) {
        m->p[i] = global[i] + (m->p[i] - global[i]) * scale;
    }
    return scale;
}
