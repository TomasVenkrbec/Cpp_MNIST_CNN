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
};

#endif