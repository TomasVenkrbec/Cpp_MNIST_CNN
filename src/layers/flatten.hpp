#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "../layer.hpp"

class FlattenLayer: public Layer {
private:
    unsigned int kernel_size;

public:
    /**
     * @brief Construct a FlattenLayer object with selected kernel size
     */
    FlattenLayer();

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