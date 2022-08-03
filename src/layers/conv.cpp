#include <iostream>
#include <vector>
#include <random>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "../activation.hpp"
#include "../initializer.hpp"
#include "conv.hpp"

using namespace std;

Conv2D::Conv2D(unsigned int kernel_size, unsigned int kernel_count, Activation* activation, bool padding, Initializer* initializer) : Layer(kernel_size * kernel_size * kernel_count) {
    this->name = "Conv2D";
    this->kernel_count = kernel_count;
    this->kernel_size = kernel_size;
    this->padding = padding;
    this->activation = activation;
    this->initializer = initializer;
    this->process_by_channel = false;
}

void Conv2D::calculate_output_shape(unsigned int input_shape[3]) {
    if (!padding && (input_shape[0] < this->kernel_size || input_shape[1] < this->kernel_size)) {
        cerr << "ERROR: Image size (" << input_shape[0] << "," << input_shape[1] << ") is smaller than kernel size (" << this->kernel_size << "," << this->kernel_size << ")." << endl;
        throw;
    }

    if (padding) { // With padding, resolution stays the same
        this->output_shape[0] = input_shape[0];
        this->output_shape[1] = input_shape[1];
    }
    else { // Without it, it decreases
        this->output_shape[0] = input_shape[0] - (this->kernel_size - 1);
        this->output_shape[1] = input_shape[1] - (this->kernel_size - 1);
    }

    // Number of output feature maps == number of filters 
    this->output_shape[2] = this->kernel_count;
}

vector<Matrix*> Conv2D::process_sample(vector<Matrix*> sample) {
    // Create matrix for result, based on pre-calculated output shape, which already counts with padding
    vector<Matrix*> res_sample;
    Matrix* result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);

    // Get kernels from neurons
    vector<float> kernel_vector;
    vector<Matrix*> kernel_matrices;
    vector<float> kernel_biases;

    for (auto a: this->get_neurons()) {
        if (kernel_vector.size() == 0) { // First neuron of kernel has the bias
            kernel_biases.push_back(a->bias);
        }

        kernel_vector.push_back(a->weights[0]); // CNN neurons have only one weight

        if (kernel_vector.size() == this->kernel_size * this->kernel_size) { // If we have all weights from filter, save it and move to next one
            Matrix* kernel = new Matrix(this->kernel_size, this->kernel_size);
            kernel->set_matrix_from_vector(kernel_vector);
            kernel_matrices.push_back(kernel);
            kernel_vector.clear();
        } 
    }

    int start_x, start_y;
    if (this->padding) {
        start_x = start_y = -(this->kernel_size - 1); // Add padding
    }
    else {
        start_x = start_y = 0; // Start at beginning
    }

    for(unsigned int i = 0; i < kernel_matrices.size(); i++) { // Iterate over CNN kernels
        for(unsigned int j = 0; j < sample.size(); j++) { // Iterate over input channels
            for (int x = 0; x < this->output_shape[0]; x++) { // Iterate over input rows accordingly to pre-calculated output shape
                for (int y = 0; y < this->output_shape[1]; y++) { // Iterate over input cols accordingly to pre-calculated output shape
                    float result = sample[j]->convolve(kernel_matrices[i], x + start_x, y + start_y); // Get result of convolution
                    result += kernel_biases[i]; // Add bias (which is the same for all neurons from kernel)
                    result = this->activation->call(result); // Perform call of activation function
                    
                    result_matrix->set_matrix(x, y, result_matrix->at(x, y) + result); // Add the value to matrix
                }
            }
        }
        // Kernel is done, save result and reset the matrix for next kernel
        res_sample.push_back(result_matrix);
        result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);
    }

    return res_sample;
}

void Conv2D::initialize_neurons() {
    this->trainable_weights_count = 0; // Reset the counter of trainable weights

    for (unsigned int i = 0; i < this->neurons.size(); i++) { // Iterate over neurons from current layer
        // Since there's one bias per kernel, the first neuron will contain the bias for entire kernel
        if (i % (this->kernel_size * this->kernel_size) == 0) {
            // Initialize bias randomly
            this->neurons[i]->bias = this->initializer->call();
            this->trainable_weights_count++;
        }

        // The number of kernel weights does not depend on previous layer - one weight per neuron
        this->neurons[i]->weights.push_back(this->initializer->call()); // Initialize randomly
        this->neurons[i]->weights_d.push_back(0.0); // Default value
        this->trainable_weights_count++;
    }
}