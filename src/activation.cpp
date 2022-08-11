#include <iostream>
#include "activation.hpp"

using namespace std;

Activation::Activation() {

}

float Activation::call(float u) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

float Activation::get_derivative(float activation) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}