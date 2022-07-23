#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "flattenlayer.hpp"

using namespace std;

FlattenLayer::FlattenLayer() : Layer(0) {
    this->name = "FlattenLayer";
}

void FlattenLayer::calculate_output_shape(unsigned int input_shape[3]) {
    if (input_shape[1] == 0 || input_shape[2] == 0) {
        cerr << "ERROR: Can't use Flatten on input with size (" << input_shape[0] << "," << input_shape[1] << "), because the input is already flat" << endl;
        throw;
    }

    this->output_shape[0] = input_shape[0] * input_shape[1] * input_shape[2];
    this->output_shape[1] = 0;
    this->output_shape[2] = 0;
}   

void FlattenLayer::forward() {
    cout << "FlattenLayer::forward()" << endl;
}