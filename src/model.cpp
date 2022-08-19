#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "aliases.hpp"
#include "model.hpp"
#include "layer.hpp"
#include "callback.hpp"
#include "utils.hpp"
#include "regularizer.hpp"

using namespace std;

void Model::add_layer(Layer* layer) {
    if (this->input_layer == NULL) {
        this->input_layer = layer;
        this->output_layer = layer;
    } 
    else {
        this->output_layer->add_next_layer(layer);
        layer->add_prev_layer(this->output_layer);
        this->output_layer = layer;
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
    unsigned int neuron_count = 0;
    Layer *cur_layer = this->input_layer;
    cout << "Network structure:" << endl;
    cout << "Input size: (" << this->dataset->get_resolution() << "," << this->dataset->get_resolution() << "," << this->dataset->get_channels() << ")" << endl; 
    while (cur_layer != NULL) {
        cout << "Layer " << ++layer_count << ": " << cur_layer->name << endl;
        if (cur_layer->get_neuron_count() > 0) {
            cout << "\t Neuron count: " << cur_layer->get_neuron_count() << endl;
            neuron_count += cur_layer->get_neuron_count();
            cout << "\t Trainable weights: " << cur_layer->get_trainable_weights_count() << endl;
            trainable_weights_count += cur_layer->get_trainable_weights_count();
        }
        if (cur_layer->get_activation() != NULL) {
            cout << "\t Activation: " << cur_layer->get_activation()->name << endl;
        }
        cout << "\t Output shape: (" << cur_layer->get_output_shape()[0] << "," << cur_layer->get_output_shape()[1] << "," << cur_layer->get_output_shape()[2] << ")" << endl;
        cur_layer = cur_layer->get_next_layer();
    }

    cout << "Total number of neurons: " << neuron_count << endl;
    cout << "Total number of trainable weights: " << trainable_weights_count << endl;
}

void Model::compile(DatasetLoader* dataset, Loss* loss, Optimizer* optimizer, Regularizer* regularizer, vector<Callback*> callbacks) {
    this->dataset = dataset;
    this->loss = loss;
    this->optimizer = optimizer;
    this->regularizer = regularizer;
    this->callbacks = callbacks;

    // Setup all layers to have their correct inputs and outputs
    unsigned int input_shape[3] = {this->dataset->get_resolution(), this->dataset->get_resolution(), this->dataset->get_channels()};
    this->input_shape = input_shape;

    if (this->input_layer == NULL) {
        cerr << "ERROR: Can't compile model without any layers." << endl;
        throw;
    }

    // Pre-calculate output shapes for all layers and initialize their neurons
    Layer* cur_layer = this->input_layer;
    unsigned int* output_shape = this->input_shape; // Pretend that the sample is output from non-existent layer
    while (cur_layer != NULL) { // Go through the network and calculate output shapes
        cur_layer->calculate_output_shape(output_shape);
        output_shape = cur_layer->get_output_shape(); // Get output shape as input shape for next layer

        cur_layer->batch_size = this->dataset->batch_size;
        cur_layer->initialize_neurons();

        cur_layer = cur_layer->get_next_layer();
    }
}

void Model::reset_callbacks() {
    for (auto callback: this->callbacks) {
        callback->reset();
    }
    loss->reset();
}

Batch Model::forward_pass(Batch data) {
    // Pass input data to input layer and results to next layers
    Layer* cur_layer = this->input_layer;
    while (cur_layer != NULL) {
        data = cur_layer->forward(data);
        cur_layer = cur_layer->get_next_layer();
    }

    return data;
}

void Model::update_weights() {
    Layer* cur_layer = this->input_layer;
    while (cur_layer != NULL) {
        vector<Neuron*> cur_layer_neurons = cur_layer->get_neurons();
        for (unsigned int i = 0; i < cur_layer_neurons.size(); i++) { // All neurons from current layer
            // Calculate regularization factors, if specified
            float bias_reg_factor = 0.0;
            float weights_reg_factor = 0.0;
            if (this->regularizer != NULL) {
                bias_reg_factor = this->regularizer->get_bias_penalty(cur_layer);
                weights_reg_factor = this->regularizer->get_weight_penalty(cur_layer);
            }

            // Update bias
            cur_layer_neurons[i]->bias += this->optimizer->call((cur_layer_neurons[i]->bias_g + bias_reg_factor) / this->dataset->batch_size);
            // Clear derivative
            cur_layer_neurons[i]->bias_g = 0.0;

            // Update weights
            for (unsigned int j = 0; j < cur_layer_neurons[i]->weights.size(); j++) { // All weights from current neuron
                cur_layer_neurons[i]->weights[j] += this->optimizer->call((cur_layer_neurons[i]->weights_g[j] + weights_reg_factor) / this->dataset->batch_size);
                // Clear derivative
                cur_layer_neurons[i]->weights_g[j] = 0.0;
            }

            // Clear neuron derivative
            cur_layer_neurons[i]->derivative = 0.0;
        }

        // Get next layer
        cur_layer = cur_layer->get_next_layer();
    }
}

void Model::backprop(Batch y_pred, Batch y_true) {
    vector<vector<float>> loss_derivatives_batch = this->loss->get_derivatives(y_pred, y_true); // Get derivatives for last layer

    for (unsigned int i = 0; i < y_pred.size(); i++) { // For every sample from batch
        Layer* cur_layer = this->output_layer; // Start from last layer
        vector<float> loss_derivatives = loss_derivatives_batch[i]; // Get loss derivatives for current sample
        
        // Calculate derivative of last layer activation given the loss derivative
        vector<Neuron*> cur_layer_neurons = cur_layer->get_neurons();
        for (unsigned int j = 0; j < cur_layer_neurons.size(); j++) { 
            cur_layer_neurons[j]->derivative = loss_derivatives[j]; // Calculate total derivative and save it
        }

        // Calculate derivatives of weights and biases for all the layers
        while(cur_layer->get_prev_layer() != NULL) { // End at input layer
            // Calculate derivatives of neuron activation
            cur_layer->add_activation_derivatives(i);

            // Calculate derivative for bias of neurons from current layer
            cur_layer->add_bias_derivatives();

            // Calculate derivatives for all weights of neurons from current layer
            cur_layer->add_weight_derivatives(i);
        
            // Calculate activation derivatives for all neurons from previous layer
            cur_layer->add_prev_layer_derivatives();

            // Clear derivatives of current layer, as they were already used for calculation
            cur_layer->clear_layer_derivatives();

            // Move one layer back
            cur_layer = cur_layer->get_prev_layer();
        }
    }
}

void Model::step(vector<DataSample*> batch_data, bool is_training) {
    // Get data and labels from DataSample
    Batch data;
    LabelsScalar labels;
    parse_datasample(batch_data, &data, &labels);
    LabelsOneHot labels_gt = one_hot(labels, this->dataset->get_max_label() + 1);
    
    // Forward pass
    LabelsOneHot labels_pred = this->forward_pass(data);

    // Call callbacks
    for (auto callback: this->callbacks) {
        callback->call(labels_pred, labels_gt);
        float cb_res;
        if (is_training) { // During training, get moving average
            cb_res = callback->get_moving_avg();
        }
        else { // During validation, prefer average of all samples
            cb_res = callback->get_epoch_avg();
        }
        cout << ", " << callback->name << ": " << fixed << setprecision(3) << cb_res;
    }

    // Calculate loss
    this->loss->call(labels_pred, labels_gt);
    float loss;
    if (is_training) { // During training, get moving average
        loss = this->loss->get_moving_avg();
    }
    else { // During validation, prefer average of all samples
        loss = this->loss->get_epoch_avg();
    }
    cout << ", Loss: " << fixed << setprecision(3) << loss;

    if (is_training) { // Don't learn during validation
        // Perform backpropagation
        this->backprop(labels_pred, labels_gt);

        // Update weights
        this->update_weights();
    }

    // Clear ground truth labels and predicted labels from memory 
    for (unsigned int i = 0; i < labels_gt.size(); i++) { // Iterate over samples
        for (unsigned int j = 0; j < labels_gt[i].size(); j++) { // Iterate over individual label values
            delete labels_gt[i][j];
            delete labels_pred[i][j];
        }
    }
}

void Model::train() {
    unsigned int steps_per_epoch = floor(this->dataset->get_train_sample_count() / this->dataset->batch_size);
    this->reset_callbacks();
    this->dataset->reset_train_batch_generator(); // Reset training batch generator
    for (unsigned int step = 1; step <= steps_per_epoch; step++) { 
        cout << "Step: " << step << "/" << steps_per_epoch;

        // Get batch of training data
        vector<DataSample*> batch_data = this->dataset->get_train_batch();
        
        // Perform training step
        this->step(batch_data, true);

        cout.flush();
        cout << "\t\t\r"; // Move further in line (to clear everything properly) and move cursor to beginning of line
    }
}

void Model::validate() {
    unsigned int steps_count = floor(this->dataset->get_val_sample_count() / this->dataset->batch_size);

    this->reset_callbacks();
    this->dataset->reset_val_batch_generator(); // Reset validation batch generator
    for (unsigned int step = 1; step <= steps_count; step++) { 
        cout << "Validation step: " << step << "/" << steps_count;

        // Get batch of validation data
        vector<DataSample*> batch_data = this->dataset->get_val_batch();

        // Perform validation step
        this->step(batch_data, false);

        cout.flush();
        cout << "\t\t\r"; // Move further in line (to clear everything properly) and move cursor to beginning of line
    }
}

void Model::fit(unsigned int max_epochs) {
    for (unsigned int epoch = 1; epoch <= max_epochs; epoch++) {
        cout << endl << "Epoch: " << epoch << "/" << max_epochs << endl;
        this->train(); // Perform training for 1 epoch
        cout << endl; // New line to not delete the training progress bar
        this->validate(); // Perform validation
        cout << endl;
    }
}