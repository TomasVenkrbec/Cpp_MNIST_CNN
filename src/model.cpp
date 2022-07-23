#include <iostream>
#include "model.hpp"
#include "layer.hpp"

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

void Model::print_model() {
    cout << endl;
    cout << "Loss: " << this->loss->name << endl;
    cout << "Optimizer: " << this->optimizer->name << endl << endl;


    unsigned int layer_count = 0;
    Layer *cur_layer = this->input_layer;
    cout << "Network structure:" << endl;
    cout << "Input size: (" << this->dataset->get_resolution() << "," << this->dataset->get_resolution() << "," << this->dataset->get_channels() << ")" << endl; 
    while (cur_layer != NULL) {
        cout << "Layer " << ++layer_count << ": " << cur_layer->name << endl;
        cout << "\t Neuron count: " << cur_layer->get_neuron_count() << endl;
        cout << "\t Output shape: (" << cur_layer->get_output_shape()[0] << "," << cur_layer->get_output_shape()[1] << "," << cur_layer->get_output_shape()[2] << ")" << endl;
        cur_layer = cur_layer->get_next_layer();
    }
}

void Model::compile(DatasetLoader* dataset, Loss* loss, Optimizer* optimizer) {
    this->dataset = dataset;
    this->loss = loss;
    this->optimizer = optimizer;

    // Setup all layers to have their correct inputs and outputs
    unsigned int input_shape[3] = {this->dataset->get_resolution(), this->dataset->get_resolution(), this->dataset->get_channels()};

    if (this->input_layer == NULL) {
        cerr << "ERROR: Can't compile model without any layers." << endl;
        throw;
    }

    // Pre-calculate output shapes for all layers
    Layer* cur_layer = this->input_layer;
    unsigned int* output_shape = input_shape; // Pretend that the sample is output from non-existent layer
    while (cur_layer != NULL) { // Go through the network and calculate output shapes
        cur_layer->calculate_output_shape(output_shape);
        output_shape = cur_layer->get_output_shape();
        cur_layer = cur_layer->get_next_layer();
    }

    // Initialize neurons in all layers
}