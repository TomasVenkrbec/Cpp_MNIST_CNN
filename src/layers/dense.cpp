#include <iostream>
#include <vector>
#include "../aliases.hpp"
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "../activation.hpp"
#include "../initializer.hpp"
#include "dense.hpp"

using namespace std;

Dense::Dense(unsigned int neuron_count, Activation* activation, Initializer* initializer) : Layer(neuron_count) {
    this->name = "Dense";
    this->activation = activation;
    this->initializer = initializer;
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

    for (unsigned int i = 0; i < neurons.size(); i++) { // Iterate over all neurons
        float result = 0.0;
        for(unsigned int j = 0; j < data->get_x_size(); j++) { // Iterate over all neurons from previous layer
            result += neurons[i]->weights[j] * data->at(j, 0); // Multiply data by corresponding weight
        }

        result += neurons[i]->bias; // Add bias
        result = this->activation->call(result); // Call activation function
        
        // Save the value
        neurons[i]->activation.push_back(result);
        result_matrix->set_matrix(i, 0, result); 
    }
    
    // Clear input data from memory
    delete data;

    return result_matrix;
}

void Dense::initialize_neurons() {
    this->trainable_weights_count = 0; // Reset the counter of trainable weights

    // Get output shape of previous layer
    unsigned int* input_shape = this->prev_layer->get_output_shape();

    for (unsigned int i = 0; i < this->neurons.size(); i++) { // Iterate over neurons from current layer
        for (unsigned int j = 0; j < input_shape[0]; j++) { // Iterate over neurons from previous layer
            // Initialize weights and their derivatives - one for each neuron in previous layer
            this->neurons[i]->weights.push_back(this->initializer->call()); // Random number
            this->neurons[i]->weights_g.push_back(0.0); // Default value
            this->trainable_weights_count++;
        }

        // Initialize bias randomly
        this->neurons[i]->bias = this->initializer->call();
        this->trainable_weights_count++;
    }
}

float Dense::get_activation_derivative(float activation) {
    return this->activation->get_derivative(activation);
}