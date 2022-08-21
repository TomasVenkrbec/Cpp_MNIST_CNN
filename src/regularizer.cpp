#include <iostream>
#include "regularizer.hpp"

using namespace std;

Regularizer::Regularizer() {

}

Regularizer::~Regularizer() {

}

float Regularizer::get_weight_penalty(Layer* layer) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

float Regularizer::get_bias_penalty(Layer* layer) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}