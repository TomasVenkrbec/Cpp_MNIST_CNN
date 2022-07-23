#include <iostream>
#include <vector>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "../activation.hpp"
#include "denselayer.hpp"

using namespace std;

DenseLayer::DenseLayer(unsigned int neuron_count, Activation *activation) : Layer(neuron_count) {
    this->name = "DenseLayer";
    this->activation = activation;
}

void DenseLayer::forward(unsigned int input_shape[4], vector<vector<Matrix*>> data) {
    // Data structures for results
    vector<vector<Matrix*>> res_samples;
    vector<Matrix*> res_sample;
    Matrix* res_channel;

    vector<Neuron> neurons = this->get_neurons();

    for (unsigned int i = 0; i < data.size(); i++) { // Iterate over samples from batch
    
    }
}

void DenseLayer::calculate_output_shape(unsigned int input_shape[3]) {
    this->output_shape[0] = this->get_neuron_count();
    this->output_shape[1] = 0;
    this->output_shape[2] = 0;
}