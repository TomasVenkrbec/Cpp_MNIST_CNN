#include "relu.hpp"

ReLU::ReLU() : Activation() {
    this->name = "ReLU";
}

float ReLU::call(float u) {
    if (u < 0) {
        return 0.0;
    }
    else {
        return u;
    }
} 