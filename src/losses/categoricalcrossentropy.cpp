#include "categoricalcrossentropy.hpp"
#include "../aliases.hpp"
#include "../utils.hpp"

CategoricalCrossentropy::CategoricalCrossentropy(unsigned int moving_average_samples) : Loss() {
    this->name = "CategoricalCrossentropy";
    this->moving_average = new MovingAverage(moving_average_samples);
}

float CategoricalCrossentropy::call(Batch y_pred, Batch y_true) {
    return 0.0;
}