#include <iostream>
#include <cmath>
#include "categoricalcrossentropy.hpp"
#include "../aliases.hpp"
#include "../utils.hpp"
#include "../layers/softmax.hpp"

using namespace std;

CategoricalCrossentropy::CategoricalCrossentropy(unsigned int moving_average_samples) : Loss() {
    this->name = "CategoricalCrossentropy";
    this->moving_average = new MovingAverage(moving_average_samples);
}

float CategoricalCrossentropy::call(LabelsOneHot y_pred, LabelsOneHot y_true) {
    float total_loss = 0.0;
    for (unsigned int i = 0; i < y_pred.size(); i++) { // Iterate over samples
        float sample_loss = 0.0;
        // Cross-entropy = -sum(y_true * log(y_pred))
        for (unsigned int j = 0; j < y_pred[i][0]->get_x_size(); j++) { // Number of rows in first channel of data (there shouldn't be any more channels) 
            sample_loss += y_true[i][0]->at(j, 0) * log2(y_pred[i][0]->at(j, 0)); // j-th row - individual probabilities, first column (there shouldn't be more)
        }
        sample_loss = -sample_loss; // Take negative to get positive final value
        total_loss += sample_loss; // Add to total
    }

    // Get average loss over entire batch
    total_loss = total_loss / y_pred.size();

    // Add towards epoch sum
    this->epoch_count++;
    this->epoch_sum += total_loss;

    // Add towards moving average
    this->moving_average->add(total_loss);

    return this->moving_average->get();
}