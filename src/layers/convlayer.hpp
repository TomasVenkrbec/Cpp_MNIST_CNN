#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "../layer.hpp"

class ConvLayer: public Layer {
private:
    unsigned int kernel_size;
    unsigned int kernel_count;
    bool padding;

public:
    /**
     * @brief Construct a ConvLayer object with selected kernel size and count
     * 
     * @param kernel_size Size of kernel (kernel_size * kernel_size)
     * @param kernel_count Kernel (filter) count in layer
     */
    ConvLayer(unsigned int kernel_size, unsigned int kernel_count, bool padding = true);

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