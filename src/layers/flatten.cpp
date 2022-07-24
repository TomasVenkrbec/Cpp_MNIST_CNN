#include <iostream>
#include <vector>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "../matrix.hpp"
#include "flatten.hpp"

using namespace std;

Flatten::Flatten() : Layer(0) {
    this->name = "Flatten";
    this->process_by_channel = false;
}

void Flatten::calculate_output_shape(unsigned int input_shape[3]) {
    if (input_shape[1] == 0 || input_shape[2] == 0) {
        cerr << "ERROR: Can't use Flatten on input with size (" << input_shape[0] << "," << input_shape[1] << "), because the input is already flat" << endl;
        throw;
    }

    this->output_shape[0] = input_shape[0] * input_shape[1] * input_shape[2];
    this->output_shape[1] = 1;
    this->output_shape[2] = 1;
}   

vector<Matrix*> Flatten::process_sample(vector<Matrix*> sample) {
    // Create data structures for result, based on pre-calculated output shape
    vector<Matrix*> res_sample;
    Matrix* result_matrix = new Matrix(this->output_shape[0], this->output_shape[1]);

    unsigned int sample_count = 0;
    for (unsigned int i = 0; i < sample.size(); i++) { // Iterate over channels
        for (unsigned int j = 0; j < sample[i]->get_x_size(); j++) { // Iterate over rows of channel
            for (unsigned int k = 0; k < sample[i]->get_x_size(); k++) { // Iterate over cols of channel
                result_matrix->set_matrix(sample_count++, 0, sample[i]->at(j, k)); // Save corresponding value to result matrix
            }
        }
    }

    if (sample_count != this->output_shape[0]) {
        cerr << "ERROR: Sample count (" << sample_count << ") doesn't match the expected output size (" << this->output_shape[0] << ")" << endl;
        throw;
    }

    res_sample.push_back(result_matrix);
    return res_sample;
}