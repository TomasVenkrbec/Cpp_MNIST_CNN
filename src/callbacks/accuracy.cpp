#include <iostream>
#include "accuracy.hpp"
#include "../callback.hpp"
#include "../matrix.hpp"
#include "../utils.hpp"
#include "../aliases.hpp"

using namespace std;

Accuracy::Accuracy(unsigned int moving_average_samples) : Callback() {
    this->name = "Accuracy";
    this->moving_average_samples = moving_average_samples;
}

float Accuracy::call(Batch y_pred, Batch y_true) {
    unsigned int hits = 0; // Correct prediction count
    
    for(unsigned int i = 0; i < y_pred.size(); i++) { // Iterate over samples
        if (get_argmax_pred(y_pred[i]) == get_argmax_pred(y_true[i])) {
            hits++;
        }
    }

    float accuracy = (float) hits / (float) y_pred.size();

    // Add towards epoch sum
    this->epoch_count++;
    this->epoch_sum += accuracy;

    // Add towards moving average
    this->moving_average_add(accuracy);

    return this->moving_average_get();
}

void Accuracy::reset() {
    this->moving_average_reset();
    this->epoch_count = 0;
    this->epoch_sum = 0.0;
}

float Accuracy::get_epoch_avg() {
    return this->epoch_sum / this->epoch_count;
}