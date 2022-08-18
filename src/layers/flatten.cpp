#include <iostream>
#include "../aliases.hpp"
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "flatten.hpp"

using namespace std;

Flatten::Flatten() : Layer(0) {
    this->name = "Flatten";
    this->process_by_channel = false;
}

void Flatten::calculate_output_shape(unsigned int input_shape[3]) {
    if (input_shape[1] == 0 || input_shape[2] == 0) {
        cerr << "ERROR: Can't use Flatten on input with size (" << input_shape[0] << "," << input_shape[1] << "), because the input is already flat" << endl;
        throw;
    }

    this->output_shape[0] = input_shape[0] * input_shape[1] * input_shape[2];
    this->output_shape[1] = 1;
    this->output_shape[2] = 1;
}   

Sample Flatten::process_sample(Sample sample) {
    // Create data structures for result, based on pre-calculated output shape
    Sample res_sample;
    Matrix* result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);

    unsigned int sample_count = 0;
    for (unsigned int i = 0; i < sample.size(); i++) { // Iterate over channels
        for (unsigned int j = 0; j < sample[i]->get_x_size(); j++) { // Iterate over rows of channel
            for (unsigned int k = 0; k < sample[i]->get_x_size(); k++) { // Iterate over cols of channel
                result_matrix->set_matrix(sample_count, 0, sample[i]->at(j, k)); // Save corresponding value to result 
                this->neurons[sample_count]->activation.push_back(sample[i]->at(j, k)); // Save to neuron
                sample_count++;
            }
        }
    }

    if (sample_count != this->output_shape[0]) {
        cerr << "ERROR: Sample count (" << sample_count << ") doesn't match the expected output size (" << this->output_shape[0] << ")" << endl;
        throw;
    }

    res_sample.push_back(result_matrix);
    return res_sample;
}

void Flatten::initialize_neurons() {
    // Since the shape depends entirely on last layer, the neurons will be added now instead of in base class constructor
    for (unsigned int i = 0; i < this->output_shape[0] * this->output_shape[1] * this->output_shape[2]; i++) {
        Neuron *neuron = new Neuron;
        this->neurons.push_back(neuron);
    }
    
    // Since the layer has no learnable parameters, there's no need to further initialization of the neurons
}

float Flatten::get_activation_derivative(float activation) {
    return activation; // Since input is only rescaled, the derivative doesn't change
}