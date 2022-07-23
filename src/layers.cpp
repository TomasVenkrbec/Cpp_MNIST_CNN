#include <iostream>
#include "layers.hpp"
#include "neuron.hpp"

using namespace std;

Layer::Layer(unsigned int neuron_count) {
    cout << "Initializing layer: " << this->name << endl;
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

ConvLayer::ConvLayer(unsigned int kernel_size, unsigned int kernel_count) : Layer(kernel_size * kernel_size * kernel_count) {

}