#include "../layer.hpp"
#include "../matrix.hpp"
#include "input.hpp"

using namespace std;

Input::Input() : Layer(0) {
    this->name = "Input";
    this->save_results = true;
}

void Input::calculate_output_shape(unsigned int input_shape[3]) {
    this->output_shape[0] = input_shape[0];
    this->output_shape[1] = input_shape[1];
    this->output_shape[2] = input_shape[2];
}

Matrix* Input::process_channel(Matrix* data) {
    return data;
}

void Input::initialize_neurons() {
    // Since the shape depends entirely on last layer, the neurons will be added now instead of in base class constructor
    for (unsigned int i = 0; i < this->output_shape[0] * this->output_shape[1] * this->output_shape[2]; i++) {
        Neuron *neuron = new Neuron;
        this->neurons.push_back(neuron);
    }
    
    // Since the layer has no learnable parameters, there's no need to further initialization of the neurons
}