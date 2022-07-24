#include <iostream>
#include <vector>
#include "accuracy.hpp"
#include "../callback.hpp"
#include "../matrix.hpp"
#include "../utils.hpp"

using namespace std;

Accuracy::Accuracy() : Callback() {
    this->name = "Accuracy";
}

float Accuracy::call(vector<vector<Matrix*>> y_pred, vector<vector<Matrix*>> y_true) {
    unsigned int hits = 0; // Correct prediction count
    
    for(unsigned int i = 0; i < y_pred.size(); i++) { // Iterate over samples
        if (get_argmax_pred(y_pred[i]) == get_argmax_pred(y_true[i])) {
            hits++;
        }
    }

    return (float) hits / (float) y_pred.size();
}