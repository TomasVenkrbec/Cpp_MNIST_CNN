#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "avgpool.hpp"

using namespace std;

AvgPoolLayer::AvgPoolLayer(unsigned int kernel_size) : Layer(0) {
    this->kernel_size = kernel_size;
    this->name = "AvgPoolLayer";
}

void AvgPoolLayer::calculate_output_shape(unsigned int input_shape[3]) {
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

void AvgPoolLayer::forward() {
    cout << "AvgPoolLayer::forward()" << endl;
}