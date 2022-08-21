#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <string>

class Activation {
public:
    std::string name;

    /**
     * @brief Activation object constructor
     */
    Activation();

    /**
     * @brief Activation object destructor
     */
    virtual ~Activation();

    /**
     * @brief Perform activation calculation
     * 
     * @param u Value of neuron
     * @return Activation of neuron
     */
    virtual float call(float u);

    /**
     * @brief Get the derivation of activation function wrt given activation
     * 
     * @param activation Activation value
     * @return Activation function derivative
     */
    virtual float get_derivative(float activation);
};

#endif