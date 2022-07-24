#include <cmath>
#include "sigmoid.hpp"

using namespace std;

Sigmoid::Sigmoid() : Activation() {
    this->name = "Sigmoid";
}

float Sigmoid::call(float u) {
    return 1 / (1 + exp(-u));
} 