#include <iostream>
#include <vector>
#include "callback.hpp"
#include "matrix.hpp"

using namespace std;

Callback::Callback() {

}

float Callback::call(vector<vector<Matrix*>> y_pred, vector<vector<Matrix*>> y_true) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}