#include <iostream>
#include <vector>
#include <random>
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

void Dense::initialize_neurons() {
    this->trainable_weights_count = 0; // Reset the counter of trainable weights

    // Get output shape of previous layer
    unsigned int* input_shape = this->prev_layer->get_output_shape();

    // Weight initializer - normal distribution with mean = 0 and stddev = 1
    random_device rd;
    mt19937 gen(rd()); // Randomizer
    normal_distribution<float> normal(0.0, 1.0);

    for (unsigned int i = 0; i < this->neurons.size(); i++) { // Iterate over neurons from current layer
        for (unsigned int j = 0; j < input_shape[0]; j++) { // Iterate over neurons from previous layer
            // Initialize weights and their derivatives - one for each neuron in previous layer
            this->neurons[i]->weights.push_back(normal(gen)); // Random number
            this->neurons[i]->weights_d.push_back(0.0); // Default value
            this->trainable_weights_count++;
        }

        // Initialize bias randomly
        this->neurons[i]->bias = normal(gen);
        this->trainable_weights_count++;
    }
}