#include <iostream>
#include <queue>
#include "aliases.hpp"
#include "callback.hpp"
#include "matrix.hpp"

using namespace std;

Callback::Callback() {

}

void Callback::moving_average_add(float value) {
    this->moving_average_total += value; // Add value to total sum (so average can be counted with it)
    this->moving_average_queue.push(value); // Add to queue
    if (this->moving_average_queue.size() > this->moving_average_samples) { // Check if queue is full
        this->moving_average_total -= this->moving_average_queue.front(); // Subtract first element from total
        this->moving_average_queue.pop(); // Remove first element from queue
    }
}

float Callback::moving_average_get() {
    return this->moving_average_total / this->moving_average_queue.size();
}

void Callback::moving_average_reset() {
    this->moving_average_total = 0.0;
    this->moving_average_queue = queue<float> (); // Clear queue
}

float Callback::call(Batch y_pred, Batch y_true) {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}

void Callback::reset() {
    // Implemented inside derivated functions
    cerr << "ERROR: Method not implemented in derived class" << endl;
    throw;
}