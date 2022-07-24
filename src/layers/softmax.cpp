#include <iostream>
#include <vector>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "softmax.hpp"

using namespace std;

Softmax::Softmax() : Layer(0) {
    this->name = "Softmax";
    this->process_by_channel = false;
}

void Softmax::calculate_output_shape(unsigned int input_shape[3]) {
    this->output_shape[0] = input_shape[0];
    this->output_shape[1] = input_shape[1];
    this->output_shape[2] = input_shape[2];
}   

vector<Matrix*> Softmax::process_sample(vector<Matrix*> sample) {
    // Create data structures for result, based on pre-calculated output shape
    throw;
}