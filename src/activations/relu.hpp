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
};

#endif