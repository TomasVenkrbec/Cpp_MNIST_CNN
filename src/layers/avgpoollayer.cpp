#include <iostream>
#include "../layer.hpp"
#include "../neuron.hpp"
#include "avgpoollayer.hpp"

using namespace std;

AvgPoolLayer::AvgPoolLayer(unsigned int kernel_size) : Layer(0) {
    this->kernel_size = kernel_size;
    this->name = "AvgPoolLayer";
}

void AvgPoolLayer::forward() {
    cout << "AvgPoolLayer::forward()" << endl;
}