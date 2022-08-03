#ifndef CALLBACK_H
#define CALLBACK_H

#include <string>
#include <queue>
#include "aliases.hpp"
#include "matrix.hpp"

class Callback {
private:
    std::queue<float> moving_average_queue;
    float moving_average_total = 0.0;

protected:
    unsigned int moving_average_samples = 1;

    /**
     * @brief Add value to moving average queue, if the queue is full, first element is popped
     * 
     * @param value Value to be added
     */
    void moving_average_add(float value);

    /**
     * @brief Get the moving average
     * 
     * @return Moving average of callback values
     */
    float moving_average_get();

    /**
     * @brief Reset the moving average
     */
    void moving_average_reset();

public:
    std::string name;

    /**
     * @brief Callback object constructor
     */
    Callback();

    /**
     * @brief Reset the callback
     */
    virtual void reset();

    /**
     * @brief Callback caller
     * 
     * @return Accuracy of the model on given outputs and ground truth labels
     */
    virtual float call(Batch y_pred, Batch y_true);
};

#endif