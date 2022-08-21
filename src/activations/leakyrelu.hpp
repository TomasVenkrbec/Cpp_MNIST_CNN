#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "../activation.hpp"

class LeakyReLU : public Activation {
private:
    float alpha;

public:
    /**
     * @brief ReLU object constructor
     * 
     * @param alpha Size of the alpha hyperparameter
     */
    LeakyReLU(float alpha = 0.01);

    /**
     * @brief Perform activation calculation
     * 
     * @param u Value of neuron
     * @return Activation of neuron
     */
    float call(float u);

    /**
     * @brief Get the derivation of LeakyReLU function wrt given activation
     * 
     * @param activation Activation value
     * @return LeakyReLU function derivative
     */
    float get_derivative(float activation);
};

#endif