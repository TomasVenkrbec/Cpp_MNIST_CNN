#ifndef REGULARIZER_H
#define REGULARIZER_H

#include <string>
#include "layer.hpp"

class Regularizer {
public:
    std::string name;

    /**
     * @brief Regularizer object constructor
     */
    Regularizer();

    /**
     * @brief Get regularization penalty for neuron weights
     * 
     * @param layer Layer to calculate regularization penalties for
     * @return Regularization penalty for weights
     */
    virtual float get_weight_penalty(Layer* layer);

    /**
     * @brief Get regularization penalty for neuron biases
     * 
     * @param layer Layer to calculate regularization penalties for
     * @return Regularization penalty for biases
     */
    virtual float get_bias_penalty(Layer* layer);
};

#endif