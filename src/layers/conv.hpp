#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <vector>
#include "../layer.hpp"
#include "../matrix.hpp"
#include "../activation.hpp"

class Conv2D: public Layer {
private:
    unsigned int kernel_size;
    unsigned int kernel_count;
    bool padding;

public:
    /**
     * @brief Construct a Conv2D object with selected kernel size and count
     * 
     * @param kernel_size Size of kernel (kernel_size * kernel_size)
     * @param kernel_count Kernel (filter) count in layer
     */
    Conv2D(unsigned int kernel_size, unsigned int kernel_count, Activation *activation, bool padding = true);
    
    /**
     * @brief Perform convolution over given data
     * 
     * @param data Data to be convoluted
     * @return Result of convolution
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
};

#endif