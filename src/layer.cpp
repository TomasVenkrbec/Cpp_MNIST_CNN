#include <iostream>
#include <vector>
#include "layer.hpp"
#include "neuron.hpp"
#include "matrix.hpp"

using namespace std;

Layer::Layer(unsigned int neuron_count) {
    for (unsigned int i = 0; i < neuron_count; i++) {
        Neuron *neuron = new Neuron;
        this->neurons.push_back(neuron);
    }
}

Layer::~Layer() {
    this->neurons.clear();
}

unsigned int Layer::get_neuron_count() {
    return this->neurons.size();
}

 unsigned int Layer::get_trainable_weights_count() {
    return this->trainable_weights_count;
 }

unsigned int* Layer::get_output_shape() {
    return this->output_shape;
}

void Layer::add_next_layer(Layer* layer) {
    this->next_layer = layer;
}

void Layer::add_prev_layer(Layer* layer) {
    this->prev_layer = layer;
}

Layer* Layer::get_next_layer() {
    return this->next_layer;
}

Layer* Layer::get_prev_layer() {
    return this->prev_layer;
}

std::vector<Neuron*> Layer::get_neurons() {
    return this->neurons;
}

Activation* Layer::get_activation() {
    return this->activation;
}

vector<vector<Matrix*>> Layer::forward(unsigned int input_shape[4], vector<vector<Matrix*>> data) {
    // Data structures for results
    vector<vector<Matrix*>> res_samples;
    vector<Matrix*> res_sample;
    Matrix* res_channel;

    for (unsigned int i = 0; i < data.size(); i++) { // Iterate over samples from batch
        if (process_by_channel) { // Process image channel-by-channel
            for (unsigned int j = 0; j < data[i].size(); j++) { // Iterate over channels from sample
                res_channel = this->process_channel(data[i][j]); // Convolve over channel and save it

                if (res_channel->get_x_size() != this->output_shape[0] || res_channel->get_y_size() != this->output_shape[1]) {
                    cerr << "ERROR: Shape of output channel (" << res_channel->get_x_size() << "," << res_channel->get_y_size() << ") doesn't match the expected output shape (" << this->output_shape[0] << "," << this->output_shape[1] << ")" << endl;
                    throw;
                }

                res_sample.push_back(res_channel); 
            }
        }
        else { // Process image sample-by-sample
            res_sample = this->process_sample(data[i]);
        }

        if (res_sample.size() != this->output_shape[2]) {
            cerr << "ERROR: Channel count of outputs (" << res_sample.size() << ") doesn't match channel count of inputs (" << input_shape[2] << ")" << endl;
            throw;
        }

        res_samples.push_back(res_sample); // Save the result
        res_sample.clear();
    }

    if (res_samples.size() != input_shape[0]) {
        cerr << "ERROR: Batch size of outputs (" << res_samples.size() << ") doesn't match batch size of inputs (" << input_shape[0] << ")" << endl;
        throw;
    }

    return res_samples;
}

Matrix* Layer::process_channel(Matrix* channel) {
    // Implemented inside derived classes
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

vector<Matrix*> Layer::process_sample(vector<Matrix*> sample) {
    // Implemented inside derived classes
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

void Layer::calculate_output_shape(unsigned int input_shape[3]) {
    // Implemented inside derived classes    
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

void Layer::initialize_neurons() {
    // Implemented inside derived classes    
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}