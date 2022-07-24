#include <iostream>
#include <vector>
#include <cmath>
#include "model.hpp"
#include "layer.hpp"
#include "callback.hpp"

using namespace std;

void Model::add_layer(Layer* layer) {
    if (this->input_layer == NULL) {
        this->input_layer = layer;
    } 
    else {
        Layer* cur_layer = this->input_layer;

        while (cur_layer->get_next_layer() != NULL) {
            cur_layer = cur_layer->get_next_layer();
        }
        cur_layer->add_next_layer(layer);
        layer->add_prev_layer(cur_layer);
    }
}

vector<Callback*> Model::get_callbacks() {
    return this->callbacks;
}

void Model::print_model() {
    cout << endl;
    cout << "Loss: " << this->loss->name << endl;
    cout << "Optimizer: " << this->optimizer->name << endl << endl;

    unsigned int layer_count = 0;
    unsigned int trainable_weights_count = 0;
    Layer *cur_layer = this->input_layer;
    cout << "Network structure:" << endl;
    cout << "Input size: (" << this->dataset->get_resolution() << "," << this->dataset->get_resolution() << "," << this->dataset->get_channels() << ")" << endl; 
    while (cur_layer != NULL) {
        cout << "Layer " << ++layer_count << ": " << cur_layer->name << endl;
        if (cur_layer->get_neuron_count() > 0) {
            cout << "\t Neuron count: " << cur_layer->get_neuron_count() << endl;
            cout << "\t Trainable weights: " << cur_layer->get_trainable_weights_count() << endl;
            trainable_weights_count += cur_layer->get_trainable_weights_count();
        }
        if (cur_layer->get_activation() != NULL) {
            cout << "\t Activation: " << cur_layer->get_activation()->name << endl;
        }
        cout << "\t Output shape: (" << cur_layer->get_output_shape()[0] << "," << cur_layer->get_output_shape()[1] << "," << cur_layer->get_output_shape()[2] << ")" << endl;
        cur_layer = cur_layer->get_next_layer();
    }

    cout << "Total number of trainable weights: " << trainable_weights_count << endl;
}

void Model::compile(DatasetLoader* dataset, Loss* loss, Optimizer* optimizer, vector<Callback*> callbacks) {
    this->dataset = dataset;
    this->loss = loss;
    this->optimizer = optimizer;
    this->callbacks = callbacks;

    // Setup all layers to have their correct inputs and outputs
    unsigned int input_shape[3] = {this->dataset->get_resolution(), this->dataset->get_resolution(), this->dataset->get_channels()};

    if (this->input_layer == NULL) {
        cerr << "ERROR: Can't compile model without any layers." << endl;
        throw;
    }

    // Pre-calculate output shapes for all layers and initialize their neurons
    Layer* cur_layer = this->input_layer;
    unsigned int* output_shape = input_shape; // Pretend that the sample is output from non-existent layer
    while (cur_layer != NULL) { // Go through the network and calculate output shapes
        cur_layer->calculate_output_shape(output_shape);
        output_shape = cur_layer->get_output_shape(); // Get output shape as input shape for next layer

        if (cur_layer->get_neuron_count() > 0) { // If the layer has neurons, initialize them
            cur_layer->initialize_neurons();
        }

        cur_layer = cur_layer->get_next_layer();
    }
}

void Model::step() {
    // Get batch of training data
    vector<DataSample*> batch_data = this->dataset->get_train_batch();
}

void Model::validate() {
    unsigned int steps_count = floor(this->dataset->get_val_sample_count() / this->dataset->batch_size);

    this->dataset->reset_val_batch_generator(); // Reset validation batch generator
    for (unsigned int step = 1; step <= steps_count; step++) { 
        // Get batch of validation data
        vector<DataSample*> batch_data = this->dataset->get_val_batch();
    }
}

void Model::fit(unsigned int max_epochs) {
    unsigned int steps_per_epoch = floor(this->dataset->get_train_sample_count() / this->dataset->batch_size);

    for (unsigned int epoch = 1; epoch <= max_epochs; epoch++) {
        cout << "Epoch: " << epoch << "/" << max_epochs << endl;

        this->dataset->reset_train_batch_generator(); // Reset training batch generator
        for (unsigned int step = 1; step <= steps_per_epoch; step++) { 
            cout << "Step: " << step << "/" << steps_per_epoch << "\r";

            this->step(); // Perform training step
        }
        
        this->validate(); // Perform validation

        cout << endl;
    }
}