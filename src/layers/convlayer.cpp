#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "convlayer.hpp"

using namespace std;

ConvLayer::ConvLayer(unsigned int kernel_size, unsigned int kernel_count) : Layer(kernel_size * kernel_size * kernel_count) {
    this->name = "ConvLayer";
}

void ConvLayer::forward() {
    cout << "ConvLayer::forward()" << endl;
}