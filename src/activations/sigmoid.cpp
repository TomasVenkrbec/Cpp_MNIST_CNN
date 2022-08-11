#include <cmath>
#include "sigmoid.hpp"

using namespace std;

Sigmoid::Sigmoid() : Activation() {
    this->name = "Sigmoid";
}

float Sigmoid::call(float u) {
    return 1 / (1 + exp(-u));
}

float Sigmoid::get_derivative(float activation) {
    // Derivative = sigmoid(activation) * (1 - sigmoid(activation))
    return this->call(activation) * (1 - this->call(activation));
}