#include <iostream>
#include "accuracy.hpp"
#include "../callback.hpp"
#include "../matrix.hpp"
#include "../utils.hpp"
#include "../aliases.hpp"

using namespace std;

Accuracy::Accuracy(unsigned int moving_average_samples) : Callback() {
    this->name = "Accuracy";
    this->moving_average = new MovingAverage(moving_average_samples);
}

void Accuracy::call(Batch y_pred, Batch y_true) {
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
    this->moving_average->add(accuracy);
}