#ifndef ADAM_H
#define ADAM_H

#include "../optimizer.hpp"

class Adam : public Optimizer {
private:
    float beta_1;
    float beta_2;
    float epsilon;
public:
    /**
     * @brief Adam optimizer object constructor
     */
    Adam(float learning_rate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float epsilon = 1e-8);
};

#endif