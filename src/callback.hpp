#ifndef CALLBACK_H
#define CALLBACK_H

#include <string>
#include <queue>
#include "aliases.hpp"
#include "matrix.hpp"
#include "utils.hpp"

class Callback {
protected:
    unsigned int epoch_count = 0;
    float epoch_sum = 0.0;
    MovingAverage* moving_average = NULL;

public:
    std::string name;

    /**
     * @brief Callback object constructor
     */
    Callback();

    /**
     * @brief Get the average value over entire epoch
     * 
     * @return Average value during epoch
     */
    float get_epoch_avg();

    /**
     * @brief Reset the callback
     */
    void reset();

    /**
     * @brief Callback caller
     * 
     * @return Accuracy of the model on given outputs and ground truth labels
     */
    virtual float call(Batch y_pred, Batch y_true);
};

#endif