#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "../layer.hpp"

class ConvLayer: public Layer {
private:
    unsigned int kernel_size;
    unsigned int kernel_count;

public:
    /**
     * @brief Construct a ConvLayer object with selected kernel size and count
     * 
     * @param kernel_size Size of kernel (kernel_size * kernel_size)
     * @param kernel_count Kernel (filter) count in layer
     * @return Initialized ConvLayer object
     */
    ConvLayer(unsigned int kernel_size, unsigned int kernel_count);

    /**
     * @brief Forward propagation function
     */
    void forward();
};

#endif