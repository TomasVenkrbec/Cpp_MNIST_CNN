#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    float activation = 0.0;
    float bias = 0.0;
    std::vector<float> weights;
    float derivative = 0.0; // Derivative of this neuron, according to chain rule
    
    // TODO: Move some items to optimizer class - for example gradients
    float bias_g = 0.0; // Gradient of bias component
    std::vector<float> weights_g; // Gradient of weight component
};

#endif