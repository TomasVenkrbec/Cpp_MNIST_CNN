#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <vector>
#include "../layer.hpp"
#include "../activation.hpp"
#include "../activations/relu.hpp"

class Dense: public Layer {
public:
    /**
     * @brief Construct a Conv object with selected neuron count
     * 
     * @param neuron_count Layer neuron count
     */
    Dense(unsigned int neuron_count, Activation *activation);

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
};

#endif