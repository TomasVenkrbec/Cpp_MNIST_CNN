#ifndef AVGPOOLLAYER_H
#define AVGPOOLLAYER_H

#include "../layer.hpp"

class AvgPool: public Layer {
private:
    unsigned int kernel_size;

public:
    /**
     * @brief Construct a AvgPool object with selected kernel size
     * 
     * @param kernel_size Size of kernel (kernel_size * kernel_size)
     */
    AvgPool(unsigned int kernel_size);

    /**
     * @brief Perform average pooling over given data
     * 
     * @param data Data to be pooled
     * @return Result of average pooling
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