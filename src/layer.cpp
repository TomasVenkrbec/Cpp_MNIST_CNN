#include <iostream>
#include "layer.hpp"
#include "neuron.hpp"

using namespace std;

Layer::Layer(unsigned int neuron_count) {
    for (unsigned int i = 0; i < neuron_count; i++) {
        Neuron *neuron = new Neuron;
        this->neurons.push_back(*neuron);
    }
}

Layer::~Layer() {
    this->neurons.clear();
}

unsigned int Layer::get_neuron_count() {
    return this->neurons.size();
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

void Layer::forward() {}

void Layer::calculate_output_shape(unsigned int input_shape[3]) {}