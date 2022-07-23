#include "adam.hpp"
#include "../optimizer.hpp"

Adam::Adam(float learning_rate, float beta_1, float beta_2, float epsilon) : Optimizer(learning_rate) {
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->name = "Adam";
}