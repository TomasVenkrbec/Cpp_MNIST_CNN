#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <string>

class Activation {
public:
    std::string name;

    /**
     * @brief Activation object constructor
     * 
     * @param name Activation function name
     */
    Activation();

    /**
     * @brief Perform activation calculation
     * 
     * @param u Value of neuron
     * @return Activation of neuron
     */
    virtual float call(float u);
};

#endif