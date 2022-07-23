#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "../layer.hpp"

class DenseLayer: public Layer {
public:
    /**
     * @brief Construct a ConvLayer object with selected neuron count
     * 
     * @param neuron_count Layer neuron count
     */
    DenseLayer(unsigned int neuron_count);

    /**
     * @brief Forward propagation function
     */
    void forward();

    /**
     * @brief Calculate output shape of layer
     * 
     * @param input_shape Input shape ([x,y,channels])
     */
    void calculate_output_shape(unsigned int input_shape[3]); 
};

#endif