#ifndef FL_TRAIN_H
#define FL_TRAIN_H

#include <stdint.h>

#include "fl_model.h"

typedef struct {

    void (*get_sample)(void *ctx, uint32_t idx, float *x, int *y);
    void *ctx;                
    uint16_t *train_idx;
    uint32_t n_train;
    const uint16_t *val_idx;
    uint32_t n_val;
} fl_data_t;

typedef struct {
    uint32_t n;        
    uint32_t steps;    
    float loss;        
    float acc;         
} fl_result_t;

fl_result_t fl_train(fl_model_t *m, fl_data_t *d, int epochs, float lr,
                     int batch, uint32_t seed);

fl_result_t fl_eval(fl_model_t *m, const fl_data_t *d, int on_val);

float fl_clip_update(fl_model_t *m, const float *global, float max_norm);

#endif
