#include <iostream>
#include "../aliases.hpp"
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

Sample Softmax::process_sample(Sample sample) {
    float sum = 0.0;
    for (unsigned int i = 0; i < sample[0]->get_x_size(); i++) { // Iterate over rows (logits) to get total sum
        sum += sample[0]->at(i, 0); // First channel (there won't be more), i-th row, first column (there also aren't any more cols)
    }

    for (unsigned int i = 0; i < sample[0]->get_x_size(); i++) { // Iterate over rows (logits) and normalize the values using calculated sum
        sample[0]->set_matrix(i, 0, sample[0]->at(i, 0) / sum); // Normalize the value and save it
    }

    return sample;
}