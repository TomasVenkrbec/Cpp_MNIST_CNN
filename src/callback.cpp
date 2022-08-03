#include <iostream>
#include "aliases.hpp"
#include "callback.hpp"
#include "matrix.hpp"

using namespace std;

Callback::Callback() {

}

float Callback::call(Batch y_pred, Batch y_true) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}