#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "convlayer.hpp"

using namespace std;

ConvLayer::ConvLayer(unsigned int kernel_size, unsigned int kernel_count, bool padding) : Layer(kernel_size * kernel_size * kernel_count) {
    this->name = "ConvLayer";
    this->kernel_count = kernel_count;
    this->kernel_size = kernel_size;
    this->padding = padding;
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

void ConvLayer::forward() {
    cout << "ConvLayer::forward()" << endl;
}