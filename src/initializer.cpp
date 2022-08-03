#include <iostream>
#include "initializer.hpp"

using namespace std;

Initializer::Initializer() {

}

float Initializer::call() {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}