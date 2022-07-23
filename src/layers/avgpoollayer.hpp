#ifndef AVGPOOLLAYER_H
#define AVGPOOLLAYER_H

#include "../layer.hpp"

class AvgPoolLayer: public Layer {
private:
    unsigned int kernel_size;
    unsigned int kernel_count;

public:
    /**
     * @brief Construct a AvgPoolLayer object with selected kernel size
     * 
     * @param kernel_size Size of kernel (kernel_size * kernel_size)
     */
    AvgPoolLayer(unsigned int kernel_size);

    /**
     * @brief Forward propagation function
     */
    void forward();
};

#endif