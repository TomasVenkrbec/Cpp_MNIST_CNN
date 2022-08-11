#include <iostream>
#include <queue>
#include "aliases.hpp"
#include "callback.hpp"
#include "matrix.hpp"

using namespace std;

Callback::Callback() {

}

void Callback::call(Batch y_pred, Batch y_true) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

void Callback::reset() {
    if (this->moving_average != NULL) {
        this->moving_average->reset();
    }
    this->epoch_count = 0;
    this->epoch_sum = 0.0;
}

float Callback::get_epoch_avg() {
    return this->epoch_sum / this->epoch_count;
}

float Callback::get_moving_avg() {
    return this->moving_average->get();
}