#ifndef SIGMOID_H
#define SIGMOID_H

#include "../activation.hpp"

class Sigmoid : public Activation {
public:
    /**
     * @brief Sigmoid object constructor
     */
    Sigmoid();

    /**
     * @brief Perform activation calculation
     * 
     * @param u Value of neuron
     * @return Activation of neuron
     */
    float call(float u);
};

#endif