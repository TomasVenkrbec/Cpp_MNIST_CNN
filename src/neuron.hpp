#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    float activation = 0.0;
    float activation_d = 0.0; // Derivative of activation function
    float bias = 0.0;
    float bias_d = 0.0; // Derivative of bias
    std::vector<float> weights;
    std::vector<float> weights_d;
};

#endif