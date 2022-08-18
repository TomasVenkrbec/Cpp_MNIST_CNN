#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "../aliases.hpp"
#include "../layer.hpp"
#include "../activation.hpp"
#include "../activations/relu.hpp"
#include "../initializer.hpp"
#include "../initializers/randomnormal.hpp"

class Dense: public Layer {
public:
    /**
     * @brief Construct a Dense layer with selected neuron count
     * 
     * @param neuron_count Layer neuron count
     * @param activation Layer activation function
     * @param initializer Layer neuron initializer
     */
    Dense(unsigned int neuron_count, Activation* activation, Initializer* initializer = new RandomNormal());

    /**
     * @brief Perform weighted sum over given data with given weights
     * 
     * @param data Input channel
     * @return Weighted sum of channel and weights
     */
    Matrix* process_channel(Matrix* data);

    /**
     * @brief Calculate output shape of layer
     * 
     * @param input_shape Input shape ([x,y,channels])
     */
    void calculate_output_shape(unsigned int input_shape[3]);

    /**
     * @brief Initialize neurons of layer
     */
    void initialize_neurons();

    /**
     * @brief Get derivative of activation function given activation value
     * 
     * @param activation Value of activation
     * @return Activation derivative
     */
    float get_activation_derivative(float activation);
};

#endif