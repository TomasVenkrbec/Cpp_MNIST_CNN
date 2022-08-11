#ifndef RELU_H
#define RELU_H

#include "../activation.hpp"

class ReLU : public Activation {
public:
    /**
     * @brief ReLU object constructor
     */
    ReLU();

    /**
     * @brief Perform activation calculation
     * 
     * @param u Value of neuron
     * @return Activation of neuron
     */
    float call(float u);

    /**
     * @brief Get the derivation of relu function wrt given activation
     * 
     * @param activation Activation value
     * @return Relu function derivative
     */
    float get_derivative(float activation);
};

#endif