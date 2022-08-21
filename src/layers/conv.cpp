#include <iostream>
#include <random>
#include "../aliases.hpp"
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
    this->save_results = true;
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

Sample Conv2D::process_sample(Sample sample) {
    // Create matrix for result, based on pre-calculated output shape, which already counts with padding
    Sample res_sample;
    Matrix* result_matrix;

    // Get kernels from neurons
    vector<float> kernel_vector;
    Sample kernel_matrices;
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
    if (this->padding) { // Add padding
        start_x = start_y = (int) -(this->kernel_size - 1) / 2;
    }
    else { // Start at beginning
        start_x = start_y = 0; 
    }

    for (unsigned int i = 0; i < kernel_matrices.size(); i++) { // Iterate over CNN kernels
        // Reset the result matrix
        result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);

        for (unsigned int j = 0; j < sample.size(); j++) { // Iterate over input channels
            for (int x = 0; x < this->output_shape[0]; x++) { // Iterate over input rows accordingly to pre-calculated output shape
                for (int y = 0; y < this->output_shape[1]; y++) { // Iterate over input cols accordingly to pre-calculated output shape
                    float result = sample[j]->convolve(kernel_matrices[i], x + start_x, y + start_y); // Get result of convolution
                    result += kernel_biases[i]; // Add bias (which is the same for all neurons from kernel)
                    result = this->activation->call(result); // Perform call of activation function
                    
                    result_matrix->set_matrix(x, y, result_matrix->at(x, y) + result); // Add the value to matrix
                }
            }
        }
        // Kernel is done, save result
        res_sample.push_back(result_matrix);
        
        // Clear allocated memory
        delete kernel_matrices[i];
    }

    return res_sample;
}

vector<Neuron*> Conv2D::get_neurons_for_kernel(unsigned int kernel_idx) {
    vector<Neuron*> neurons;

    unsigned int kernel_start_idx = kernel_idx * this->kernel_size * this->kernel_size;
    for (unsigned int i = 0; i < this->kernel_size * this->kernel_size; i++) { // For every neuron from kernel
        neurons.push_back(this->neurons[kernel_start_idx + i]); // Get corresponding neurons
    }
    return neurons;
}

Matrix* Conv2D::get_derivatives_for_kernel(unsigned int kernel_idx) {
    Matrix* derivatives = new Matrix(this->output_shape[0], this->output_shape[1]);
    vector<float> derivatives_vector;

    unsigned int kernel_start_idx = kernel_idx * this->output_shape[0] * this->output_shape[1];
    for (unsigned int i = 0; i < this->output_shape[0] * this->output_shape[1]; i++) { // For all neurons from kernel
        derivatives_vector.push_back(this->derivatives_vector[kernel_start_idx + i]); // Get corresponding derivatives
    }
    derivatives->set_matrix_from_vector(derivatives_vector); // Save as matrix

    return derivatives;
}

float Conv2D::get_activation_derivative(float activation) {
    return this->activation->get_derivative(activation);
}

void Conv2D::add_activation_derivatives(unsigned int sample_idx) {
    for (unsigned int i = 0; i < this->derivatives_vector.size(); i++) { // Calculate activation derivative for every output pixel
        this->derivatives_vector[i] *= this->get_activation_derivative(this->derivatives_vector[i]);
    }
}

void Conv2D::add_bias_derivatives() {
    for (unsigned int i = 0; i < this->kernel_count; i++) { // For every kernel (output channel)
        Matrix* kernel_derivatives = this->get_derivatives_for_kernel(i);
        vector<Neuron*> kernel_neurons = this->get_neurons_for_kernel(i);

        // Only first neuron from kernel has the bias, add all corresponding derivatives to it
        for (unsigned int x = 0; x < kernel_derivatives->get_x_size(); x++) { // Go over rows of derivatives
            for (unsigned int y = 0; y < kernel_derivatives->get_y_size(); y++) { // Go over cols of derivatives
                kernel_neurons[0]->bias_g += kernel_derivatives->at(x, y);
            }
        }

        // Clear allocated memory
        delete kernel_derivatives;
    }
}

void Conv2D::add_weight_derivatives(unsigned int sample_idx) {
    // https://stats.stackexchange.com/questions/326377/backpropagation-on-a-convolutional-layer
    int start_x, start_y;
    if (this->padding) { // Add padding
        start_x = start_y = (int) -(this->kernel_size - 1) / 2;
    }
    else { // Start at beginning
        start_x = start_y = 0;
    }

    Sample prev_layer_output = this->prev_layer->get_saved_result()[sample_idx];
    for (unsigned int i = 0; i < this->kernel_count; i++) { // For every kernel (output channel)
        Matrix* kernel_derivatives = this->get_derivatives_for_kernel(i);
        vector<Neuron*> kernel_neurons = this->get_neurons_for_kernel(i);
        for (unsigned int j = 0; j < prev_layer_output.size(); j++) { // Iterate over input channels
            for (unsigned int x = 0; x < kernel_derivatives->get_x_size(); x++) { // Iterate over derivative matrix rows (same dimension as output)
                for (unsigned int y = 0; y < kernel_derivatives->get_y_size(); y++) { // Iterate over derivative matrix cols (same dimension as output)
                    for (unsigned int k = 0; k < kernel_neurons.size(); k++) { // Iterate over individual neurons from kernel
                        // Calculate x and y offset, wrt position of neuron in kernel and padding
                        unsigned int x_offset = (unsigned int) k / this->kernel_size + start_x;
                        unsigned int y_offset = k % this->kernel_size + start_y;

                        // Out-of-bounds coordinates can be ignored, since the result in the chain rule would be 0 anyways
                        if (x + x_offset < 0 || x + x_offset >= prev_layer_output[j]->get_x_size() || y + y_offset < 0 || y + y_offset >= prev_layer_output[j]->get_y_size()) {
                            continue;
                        }
                        
                        // Add calculated derivative to neuron's total weight derivative
                        kernel_neurons[k]->weights_g[0] += prev_layer_output[j]->at(x + x_offset, y + y_offset) * kernel_derivatives->at(x, y);
                    }
                }
            }
        }

        // Clear allocated memory
        delete kernel_derivatives;
    }
}

void Conv2D::add_prev_layer_derivatives() {
    int start_x, start_y;
    start_x = start_y = (int) -(this->kernel_size - 1) / 2;

    for (unsigned int i = 0; i < this->kernel_count; i++) { // For every kernel (output channel)
        Matrix* kernel_derivatives = this->get_derivatives_for_kernel(i);
        vector<Neuron*> kernel_neurons = this->get_neurons_for_kernel(i);
        
        // Create kernel
        Matrix* kernel = new Matrix(this->kernel_size, this->kernel_size);
        vector<float> kernel_vector;
        for (auto a: kernel_neurons) {
            kernel_vector.push_back(a->weights[0]);
        }
        kernel->set_matrix_from_vector(kernel_vector);

        for (unsigned int x = 0; x < kernel_derivatives->get_x_size(); x++) { // Iterate over derivative matrix rows (same dimension as output)
            for (unsigned int y = 0; y < kernel_derivatives->get_y_size(); y++) { // Iterate over derivative matrix cols (same dimension as output)
                float result = kernel_derivatives->convolve(kernel, x + start_x, y + start_y); // Get result of convolution
                this->prev_layer->add_layer_derivative(result);
            }
        }

        // Clear allocated memory
        delete kernel;
        delete kernel_derivatives;
    }
}

void Conv2D::initialize_neurons() {
    this->trainable_weights_count = 0; // Reset the counter of trainable weights

    for (unsigned int i = 0; i < this->neurons.size(); i++) { // Iterate over neurons from current layer
        // Since there's one bias per kernel, the first neuron will contain the bias for entire kernel
        if (i % (this->kernel_size * this->kernel_size) == 0) { // TODO: All neurons from kernel will contain a pointer to the same bias value instead of this
            // Initialize bias randomly
            this->neurons[i]->bias = this->initializer->call();
            this->trainable_weights_count++;
        }

        // The number of kernel weights does not depend on previous layer - one weight per neuron
        this->neurons[i]->weights.push_back(this->initializer->call()); // Initialize randomly
        this->neurons[i]->weights_g.push_back(0.0); // Default value
        this->trainable_weights_count++;
    }
}