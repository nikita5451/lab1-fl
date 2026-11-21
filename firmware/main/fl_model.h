#ifndef FL_MODEL_H
#define FL_MODEL_H
/*
 * fl_model — модель і один крок SGD.  НАДАЄТЬСЯ, не потрібно писати.
 *
 *     x -> W1,b1 -> ReLU -> W2,b2 -> softmax
 *
 * Конфігурація 64 -> 16 -> 10: 1210 параметрів float32 = 4840 байтів —
 * обсяг, який перетинає канал за раунд в один бік.
 *
 * РОЗКЛАДКА ПАРАМЕТРІВ (контракт моделі).  Один плоский масив float32,
 * row-major, у такому порядку:
 *
 *     W1[n_hidden][n_in], b1[n_hidden], W2[n_out][n_hidden], b2[n_out]
 *
 * Цей порядок використовується при пакуванні в байти для передачі, і його
 * повторює хостове дзеркало (host/fl_model_ref.py).  Інший порядок дає
 * різні моделі на хості й на пристрої: помилки не буде, буде нижча
 * точність.
 *
 * Жодного заголовка ESP-IDF: файл компілюється і для Xtensa, і для хоста.
 * Жодного динамічного виділення пам'яті: буфери передає викликач.
 */
#include <stddef.h>
#include <stdint.h>

#define FL_MAX_IN      64
#define FL_MAX_HIDDEN  16
#define FL_MAX_OUT     10

/* Найгірший випадок для статичних буферів: 1210 float = 4840 B. */
#define FL_MAX_PARAMS  (FL_MAX_HIDDEN * FL_MAX_IN + FL_MAX_HIDDEN + \
                        FL_MAX_OUT * FL_MAX_HIDDEN + FL_MAX_OUT)

typedef struct {
    int n_in, n_hidden, n_out;
    size_t n_params;

    float *p;     /* параметри, n_params штук — не володіє */
    float *g;     /* акумулятор градієнта    — не володіє */

    float *w1, *b1, *w2, *b2;       /* зрізи p */     
    float *gw1, *gb1, *gw2, *gb2;   /* зрізи g */   
    
    /* Робочі значення одного прямого проходу. hpre потрібне в backward:
     * похідна ReLU береться від значення ДО активації. */
    float hpre[FL_MAX_HIDDEN];
    float h[FL_MAX_HIDDEN];
    float prob[FL_MAX_OUT];
} fl_model_t;

/* Кількість параметрів для заданої форми. */
size_t fl_model_n_params(int n_in, int n_hidden, int n_out);

/* Прив'язує буфери params і grad (кожен >= fl_model_n_params float).
 * Повертає 0, або -1 якщо форма виходить за FL_MAX_*. */
int fl_model_init(fl_model_t *m, int n_in, int n_hidden, int n_out,
                  float *params, float *grad);

/* Glorot-uniform від seed, зсуви в нуль.  Детерміновано: той самий seed
 * на хості й на платі дає ті самі байти (перевірка FL-03). */
void fl_model_init_weights(fl_model_t *m, uint32_t seed);

/* Прямий прохід.  Заповнює m->prob, повертає індекс найбільшої. */
int fl_model_forward(fl_model_t *m, const float *x);

/* Перехресна ентропія останнього прямого проходу для класу y. */
float fl_model_loss(const fl_model_t *m, int y);

void fl_model_zero_grad(fl_model_t *m);

/* Прямий + зворотний прохід для ОДНОГО семпла; градієнт додається до
 * акумулятора.  Повертає втрати; *ok (можна NULL) = 1 якщо вгадано. */
float fl_model_accumulate(fl_model_t *m, const float *x, int y, int *ok);

/* p -= lr * g / batch.  Градієнт не обнуляється. */
void fl_model_sgd_step(fl_model_t *m, float lr, int batch);

/* ||p - other||_2 по всьому плоскому масиву.  Використовується, щоб
 * звітувати норму оновлення раунду проти вагів, отриманих від сервера. */
float fl_model_delta_norm(const fl_model_t *m, const float *other);

#endif /* FL_MODEL_H */
