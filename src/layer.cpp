#include <iostream>
#include "aliases.hpp"
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
    for (unsigned int i = 0; i < this->neurons.size(); i++) {
        delete this->neurons[i];
    }
    delete this->activation;
    delete this->initializer;
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

Batch Layer::get_saved_result() {
    return this->res_samples;
}

void Layer::add_layer_derivative(float derivative) {
    this->derivatives_vector.push_back(derivative);
}

Batch Layer::forward(Batch data) {
    // Data structures for results
    Batch res_samples;
    Sample res_sample;
    Matrix* res_channel;
    
    // Clear previous activations from all neurons
    for (unsigned int i = 0; i < this->neurons.size(); i++) {
        this->neurons[i]->activation.clear();
    }

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

        res_samples.push_back(res_sample); // Save the result
        res_sample.clear();
    }

    if (this->save_results) { // Save results for backpropagation purposes if needed
        this->clear_saved_samples(); // Clear previously saved samples first
        this->res_samples = res_samples;
    }

    if (res_samples.size() != data.size()) {
        cerr << "ERROR: Batch size of outputs (" << res_samples.size() << ") doesn't match batch size of inputs (" << data.size() << ")" << endl;
        throw;
    }

    return res_samples;
}

Matrix* Layer::process_channel(Matrix* channel) {
    // Implemented inside derived classes
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

Sample Layer::process_sample(Sample sample) {
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

float Layer::get_activation_derivative(float activation) {
    // Implemented inside derived classes    
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;   
}

void Layer::add_activation_derivatives(unsigned int sample_idx) {
    for (unsigned int j = 0; j < this->neurons.size(); j++) {
        this->neurons[j]->derivative *= this->get_activation_derivative(this->neurons[j]->activation[sample_idx]);
    }
}

void Layer::add_bias_derivatives() {
    for (unsigned int j = 0; j < this->neurons.size(); j++) {
        if (this->neurons[j]->weights.size() == 0) { // If the neuron has no weights, just pass through
            this->neurons[j]->bias_g += this->neurons[j]->derivative;
        }
    }
}

void Layer::add_weight_derivatives(unsigned int sample_idx) {
    for (unsigned int j = 0; j < this->neurons.size(); j++) {
        for (unsigned int k = 0; k < this->neurons[j]->weights.size(); k++) { // All weights from current neuron
            this->neurons[j]->weights_g[k] += this->prev_layer->neurons[k]->activation[sample_idx] * this->neurons[j]->derivative;
        }
    }
}

void Layer::add_prev_layer_derivatives() {
    for (unsigned int j = 0; j < this->neurons.size(); j++) {
        if (this->neurons[j]->weights.size() == 0) { // If the neuron has no weights, just pass through
            if (this->neurons.size() == this->prev_layer->neurons.size()) { // Can work only for layers doing certain transformations
                this->prev_layer->neurons[j]->derivative = this->neurons[j]->derivative;
            }
            else { // Save the derivatives to vector in previous layer object, as the layer will handle them itself
                this->prev_layer->derivatives_vector.push_back(this->neurons[j]->derivative);
            }
        }
        else { // Neuron has weights
            // Activation derivative of neuron from previous layer depends on every neuron from the current layer
            for (unsigned int k = 0; k < this->prev_layer->neurons.size(); k++) { // For every neuron of previous layer
                this->prev_layer->neurons[k]->derivative += this->neurons[j]->weights[k] * this->neurons[j]->derivative;
            }
        }
    }
}

void Layer::clear_layer_derivatives() {
    for (unsigned int j = 0; j < this->neurons.size(); j++) {
        this->neurons[j]->derivative = 0.0;
    }
    this->derivatives_vector.clear();
}

void Layer::clear_saved_samples() {
    if (this->save_results) {
        if (this->name != "Input") { // Don't clear Input layer, because it has only pointers to dataset
            for (unsigned int i = 0; i < this->res_samples.size(); i++) { // Iterate over samples
                for (unsigned int j = 0; j < this->res_samples[i].size(); j++) { // Iterate over channels
                    delete this->res_samples[i][j];
                }
            }
        }
        this->res_samples.clear();
    }
}