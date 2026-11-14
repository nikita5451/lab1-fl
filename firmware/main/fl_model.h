#ifndef FL_MODEL_H
#define FL_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define FL_MAX_IN      64
#define FL_MAX_HIDDEN  16
#define FL_MAX_OUT     10

#define FL_MAX_PARAMS  (FL_MAX_HIDDEN * FL_MAX_IN + FL_MAX_HIDDEN + \
                        FL_MAX_OUT * FL_MAX_HIDDEN + FL_MAX_OUT)

typedef struct {
    int n_in, n_hidden, n_out;
    size_t n_params;

    float *p;    
    float *g;    

    float *w1, *b1, *w2, *b2;        
    float *gw1, *gb1, *gw2, *gb2;    
    float hpre[FL_MAX_HIDDEN];
    float h[FL_MAX_HIDDEN];
    float prob[FL_MAX_OUT];
} fl_model_t;

size_t fl_model_n_params(int n_in, int n_hidden, int n_out);
int fl_model_init(fl_model_t *m, int n_in, int n_hidden, int n_out, float *params, float *grad);

void fl_model_init_weights(fl_model_t *m, uint32_t seed);
int fl_model_forward(fl_model_t *m, const float *x);
float fl_model_loss(const fl_model_t *m, int y);
void fl_model_zero_grad(fl_model_t *m);
float fl_model_accumulate(fl_model_t *m, const float *x, int y, int *ok);
void fl_model_sgd_step(fl_model_t *m, float lr, int batch);
float fl_model_delta_norm(const fl_model_t *m, const float *other);

#endif 
