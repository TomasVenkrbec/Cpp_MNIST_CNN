#include <iostream>
#include <vector>
#include "loss.hpp"
#include "aliases.hpp"
#include "callback.hpp"

using namespace std;

Loss::Loss() : Callback() {

}

void Loss::call(Batch y_pred, Batch y_true) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

vector<vector<float>> Loss::get_derivatives(Batch y_pred, Batch y_true) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}