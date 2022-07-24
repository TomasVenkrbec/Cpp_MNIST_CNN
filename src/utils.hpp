#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "matrix.hpp"

/**
 * @brief Convert batch of scalar labels to one-hot encoded labels
 * 
 * @param labels Scalar labels
 * @param label_count Count of different possible labels
 * @return Batch of one-hot encoded labels 
 */
std::vector<std::vector<Matrix*>> one_hot(std::vector<unsigned int> labels, unsigned int label_count);

/**
 * @brief Get the argmax of prediction
 * 
 * @param pred Network prediction or one-hot encoded label
 * @return Argmax of prediction
 */
unsigned int get_argmax_pred(std::vector<Matrix*> pred);

#endif