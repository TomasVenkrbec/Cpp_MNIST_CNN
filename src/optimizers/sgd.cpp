#include "sgd.hpp"
#include "../optimizer.hpp"

SGD::SGD(float learning_rate) : Optimizer(learning_rate) {
    this->name = "SGD";
}

float SGD::call(float derivative) {
    return -this->learning_rate * derivative;
}