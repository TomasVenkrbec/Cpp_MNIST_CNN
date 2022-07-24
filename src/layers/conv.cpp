#include <iostream>
#include <vector>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "../activation.hpp"
#include "conv.hpp"

using namespace std;

ConvLayer::ConvLayer(unsigned int kernel_size, unsigned int kernel_count, Activation *activation, bool padding) : Layer(kernel_size * kernel_size * kernel_count) {
    this->name = "ConvLayer";
    this->kernel_count = kernel_count;
    this->kernel_size = kernel_size;
    this->padding = padding;
    this->activation = activation;
}

void ConvLayer::calculate_output_shape(unsigned int input_shape[3]) {
    if (!padding && (input_shape[0] < this->kernel_size || input_shape[1] < this->kernel_size)) {
        cerr << "ERROR: Image size (" << input_shape[0] << "," << input_shape[1] << ") is smaller than kernel size (" << this->kernel_size << "," << this->kernel_size << ")." << endl;
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

Matrix* ConvLayer::convolve(Matrix* data) {
    // Create matrix for result, based on pre-calculated output shape, which already counts with padding
    Matrix* result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);

    // Get kernel from neurons
    vector<float> kernel_vector;
    for (auto a: this->get_neurons()) {
        kernel_vector.push_back(a.weights[0]); // CNN neurons have only one weight
    }
    Matrix *kernel = new Matrix(this->kernel_size, this->kernel_size);
    kernel->set_matrix_from_vector(kernel_vector);

    int start_x, start_y;
    if (this->padding) {
        start_x = start_y = -(this->kernel_size - 1); // Add padding
    }
    else {
        start_x = start_y = 0; // Start at beginning
    }

    for (int i = 0; i < this->output_shape[0]; i++) { // Iterate accordingly to pre-calculated output shape
        for (int j = 0; j < this->output_shape[1]; j++) { 
            float result = data->convolve(kernel, i + start_x, j + start_y);
            result_matrix->set_matrix(i, j, result);
        }
    }

    return result_matrix;
}

void ConvLayer::forward(unsigned int input_shape[4], vector<vector<Matrix*>> data) {
    // Data structures for results
    vector<vector<Matrix*>> res_samples;
    vector<Matrix*> res_sample;
    Matrix* res_channel;

    for (unsigned int i = 0; i < data.size(); i++) { // Iterate over samples from batch
        for (unsigned int j = 0; j < data[i].size(); j++) { // Iterate over channels from sample
            res_channel = this->convolve(data[i][j]); // Convolve over channel and save it

            if (res_channel->get_x_size() != this->output_shape[0] || res_channel->get_y_size() != this->output_shape[1]) {
                cerr << "ERROR: Shape of output channel (" << res_channel->get_x_size() << "," << res_channel->get_y_size() << ") doesn't match the expected output shape (" << this->output_shape[0] << "," << this->output_shape[1] << ")" << endl;
                throw;
            }

            res_sample.push_back(res_channel); 
        }

        if (res_sample.size() != this->output_shape[2]) {
            cerr << "ERROR: Channel count of outputs (" << res_sample.size() << ") doesn't match channel count of inputs (" << input_shape[2] << ")" << endl;
        }

        res_samples.push_back(res_sample); // Save the result
        res_sample.clear();
    }

    if (res_samples.size() != input_shape[0]) {
        cerr << "ERROR: Batch size of outputs (" << res_samples.size() << ") doesn't match batch size of inputs (" << input_shape[0] << ")" << endl;
        throw;
    }
}