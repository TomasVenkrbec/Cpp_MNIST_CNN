#ifndef UTILS_H
#define UTILS_H

#include <queue>
#include "aliases.hpp"
#include "matrix.hpp"
#include "dataset.hpp"

class MovingAverage {
private:
    unsigned int samples = 1;
    std::queue<float> queue;
    float total;

public:
    /**
     * @brief Construct a new MovingAverage object
     * 
     * @param samples Number of last samples to calculate the moving average from
     */
    MovingAverage(unsigned int samples) {
        this->samples = samples;
        this->total = 0.0;
    }

    /**
     * @brief Add value to moving average queue, if the queue is full, first element is popped
     * 
     * @param value Value to be added
     */
    void add(float value);

    /**
     * @brief Get the moving average
     * 
     * @return Moving average of callback values
     */
    float get();

    /**
     * @brief Reset the moving average
     */
    void reset();
};


/**
 * @brief Convert batch of scalar labels to one-hot encoded labels
 * 
 * @param labels Scalar labels
 * @param label_count Count of different possible labels
 * @return Batch of one-hot encoded labels 
 */
Batch one_hot(LabelsScalar labels, unsigned int label_count);

/**
 * @brief Get the argmax of prediction
 * 
 * @param pred Network prediction or one-hot encoded label
 * @return Argmax of prediction
 */
unsigned int get_argmax_pred(Sample pred);

/**
 * @brief Get data and label batch from batch of DataSample objects
 * 
 * @param raw_data Batch of DataSample objects
 * @param data Batch of parsed image data
 * @param labels Batch of parsed scalar labels 
 */
void parse_datasample(std::vector<DataSample*> raw_data, Batch* data, LabelsScalar* labels);

#endif