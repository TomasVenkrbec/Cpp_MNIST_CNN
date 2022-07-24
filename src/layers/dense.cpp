#include <iostream>
#include <vector>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "../activation.hpp"
#include "dense.hpp"

using namespace std;

Dense::Dense(unsigned int neuron_count, Activation *activation) : Layer(neuron_count) {
    this->name = "Dense";
    this->activation = activation;
    this->process_by_channel = true;
}

void Dense::calculate_output_shape(unsigned int input_shape[3]) {
    this->output_shape[0] = this->get_neuron_count();
    this->output_shape[1] = 1;
    this->output_shape[2] = 1;
}

Matrix* Dense::process_channel(Matrix* data) {
    // Create matrix for result, based on pre-calculated output shape
    Matrix* result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);

    vector<Neuron*> neurons = this->get_neurons();

    float result = 0.0;
    for (unsigned int i = 0; i < neurons.size(); i++) { // Iterate over all neurons
        for(unsigned int j = 0; j < data->get_x_size(); j++) { // Iterate over all neurons from previous layer
            result += neurons[i]->weights[j] * data->at(j, 0); // Multiply data by corresponding weight
        }

        result += neurons[i]->bias; // Add bias
        result = this->activation->call(result); // Call activation function
        result_matrix->set_matrix(i, 0, result); // Save the value
        result = 0.0;
    }

    return result_matrix;
}