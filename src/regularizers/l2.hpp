#ifndef L2_H
#define L2_H

#include "../regularizer.hpp"

class L2 : public Regularizer {
private:
    float reg_factor_weights;
    float reg_factor_biases;

public:
    /**
     * @brief L2 regularizer object constructor
     * 
     * @param reg_factor_weights Regularization factor of weights
     * @param reg_factor_biases Regularization factor of biases
     */
    L2(float reg_factor_weights = 1e-4, float reg_factor_biases = 1e-4);

    /**
     * @brief Get L2 regularization penalty for neuron weights
     * 
     * @param layer Layer to calculate L2 regularization penalties for
     * @return L2 regularization penalty for weights
     */
    float get_weight_penalty(Layer* layer);

    /**
     * @brief Get L2 regularization penalty for neuron biases
     * 
     * @param layer Layer to calculate L2 regularization penalties for
     * @return L2 regularization penalty for biases
     */
    float get_bias_penalty(Layer* layer);
};

#endif