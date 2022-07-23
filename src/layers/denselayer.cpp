#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "denselayer.hpp"

using namespace std;

DenseLayer::DenseLayer(unsigned int neuron_count) : Layer(neuron_count) {
    this->name = "DenseLayer";
}

void DenseLayer::forward() {
    cout << "DenseLayer::forward()" << endl;
}