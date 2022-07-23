#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "denselayer.hpp"

using namespace std;

DenseLayer::DenseLayer(unsigned int neuron_count) : Layer(neuron_count) {
    this->name = "DenseLayer";
}

void DenseLayer::forward() {
    cout << "DenseLayer::forward()" << endl;
}

void DenseLayer::calculate_output_shape(unsigned int input_shape[3]) {
    this->output_shape[0] = this->get_neuron_count();
    this->output_shape[1] = 0;
    this->output_shape[2] = 0;
}