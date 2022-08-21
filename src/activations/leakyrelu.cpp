#include "leakyrelu.hpp"

LeakyReLU::LeakyReLU(float alpha) : Activation() {
    this->name = "LeakyReLU";
    this->alpha = alpha;
}

float LeakyReLU::call(float u) {
    if (u < 0) {
        return u * this->alpha;
    }
    else {
        return u;
    }
} 

float LeakyReLU::get_derivative(float activation) {
    if (activation < 0) {
        return this->alpha;
    }
    else {
        return 1.0;
    }
}