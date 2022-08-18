#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "avgpool.hpp"

using namespace std;

AvgPool::AvgPool(unsigned int kernel_size) : Layer(0) {
    this->kernel_size = kernel_size;
    this->name = "AvgPool";
    this->process_by_channel = true;
}

void AvgPool::calculate_output_shape(unsigned int input_shape[3]) {
    if (this->kernel_size > input_shape[0] || this->kernel_size > input_shape[1]) {
        cerr << "ERROR: Can't use AvgPool with size " << this->kernel_size << " on input with size (" << input_shape[0] << "," << input_shape[1] << ")." << endl;
        throw;
    }

    if (input_shape[0] % this->kernel_size != 0 || input_shape[1] % this->kernel_size != 0) {
        cerr << "ERROR: Image size (" << input_shape[0] << "," << input_shape[1] << ") can't be pooled by factor of " << this->kernel_size << endl;
        throw;
    }

    this->output_shape[0] = (unsigned int) (input_shape[0] / this->kernel_size);
    this->output_shape[1] = (unsigned int) (input_shape[1] / this->kernel_size);
    this->output_shape[2] = input_shape[2];
}   

Matrix* AvgPool::process_channel(Matrix* data) {
    // Create matrix for result, based on pre-calculated output shape
    Matrix* result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);

    // Create averaging mask (kernel)
    Matrix *kernel = new Matrix(this->kernel_size, this->kernel_size);

    for (int i = 0; i < data->get_x_size(); i += this->kernel_size) { // Move kernel over rows from data
        for (int j = 0; j < data->get_y_size(); j += this->kernel_size) { // Move kernel over cols from data
            float result = data->get_avg(kernel, i, j); // Get average in masked area
            result_matrix->set_matrix(i / this->kernel_size, j / this->kernel_size, result); // Save the average to corresponding position
            this->neurons[i * data->get_x_size() + j]->activation.push_back(result);
        }
    }

    return result_matrix;
}

void AvgPool::initialize_neurons() {
    // Since the shape depends entirely on last layer and kernel size, the neurons will be added now instead of in base class constructor
    for (unsigned int i = 0; i < this->output_shape[0] * this->output_shape[1] * this->output_shape[2]; i++) {
        Neuron *neuron = new Neuron;
        this->neurons.push_back(neuron);
    }
    
    // Since the layer has no learnable parameters, there's no need to further initialization of the neurons
}

float AvgPool::get_activation_derivative(float activation) {
    return activation; 
}