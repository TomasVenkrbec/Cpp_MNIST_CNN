#include <iostream>
#include "loss.hpp"
#include "aliases.hpp"
#include "callback.hpp"

using namespace std;

Loss::Loss() : Callback() {

}

float Loss::call(Batch y_pred, Batch y_true) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}