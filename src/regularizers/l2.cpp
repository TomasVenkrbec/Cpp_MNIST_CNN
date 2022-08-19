#include <iostream>
#include "l2.hpp"
#include "../neuron.hpp"

using namespace std;

L2::L2(float reg_factor_weights, float reg_factor_biases) {
    this->reg_factor_weights = reg_factor_weights;
    this->reg_factor_biases = reg_factor_biases;
}

float L2::get_weight_penalty(Layer* layer) {
    vector<Neuron*> layer_neurons = layer->get_neurons();

    float weights_sum = 0.0;
    for (unsigned int i = 0; i < layer_neurons.size(); i++) {
        for (unsigned int j = 0; j < layer_neurons[i]->weights.size(); j++) {
            weights_sum += layer_neurons[i]->weights[j] * layer_neurons[i]->weights[j];
        }
    }
    return weights_sum * this->reg_factor_weights;
}

float L2::get_bias_penalty(Layer* layer) {
    vector<Neuron*> layer_neurons = layer->get_neurons();

    float bias_sum = 0.0;
    for (unsigned int i = 0; i < layer_neurons.size(); i++) {
        bias_sum += layer_neurons[i]->bias * layer_neurons[i]->bias;
    }
    return bias_sum * this->reg_factor_biases;
}
