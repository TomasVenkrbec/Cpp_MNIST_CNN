#include <iostream>
#include "optimizer.hpp"

using namespace std;

Optimizer::Optimizer(float learning_rate) {
    this->learning_rate = learning_rate;
}

Optimizer::~Optimizer() {
    
}

float Optimizer::call(float derivative) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}