#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <vector>
#include "../layer.hpp"
#include "../activation.hpp"
#include "../activations/relu.hpp"

class DenseLayer: public Layer {
public:
    /**
     * @brief Construct a ConvLayer object with selected neuron count
     * 
     * @param neuron_count Layer neuron count
     */
    DenseLayer(unsigned int neuron_count, Activation *activation);

    /**
     * @brief Forward propagation function
     *
     * @param input_shape Shape of the input ([batch_size,x,y,channels])
     * @param data Vector of vectors of channels - batch of feature maps
     */
    void forward(unsigned int input_shape[4], std::vector<std::vector<Matrix*>> data);

    /**
     * @brief Calculate output shape of layer
     * 
     * @param input_shape Input shape ([x,y,channels])
     */
    void calculate_output_shape(unsigned int input_shape[3]); 
};

#endif