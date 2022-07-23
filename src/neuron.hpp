#ifndef NEURON_H
#define NEURON_H

class Neuron {
public:
    float activation = 0.0;
    float activation_d = 0.0; // Derivative of activation function
    float bias = 0.0;
    float bias_d = 0.0;
    float u = 0.0; // Neuron potential (input of activation function)
    float* weights;
    float* weights_d;
};
#endif