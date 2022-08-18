#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "../aliases.hpp"
#include "../layer.hpp"
#include "../matrix.hpp"
#include "../activation.hpp"
#include "../initializer.hpp"
#include "../initializers/randomnormal.hpp"

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
     * @param activation Layer activation function
     * @param initializer Layer neuron initializer
     * @param padding Padding of layer
     */
    Conv2D(unsigned int kernel_size, unsigned int kernel_count, Activation *activation, bool padding = true, Initializer* initializer = new RandomNormal());
    
    /**
     * @brief Process sample with convolutional filters
     * 
     * @param sample Input sample
     * @return Resulting feature map
     */
    Sample process_sample(Sample sample);

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

    /**
     * @brief Get derivative of activation function given activation value
     * 
     * @param activation Value of activation
     * @return Activation derivative
     */
    float get_activation_derivative(float activation);
};

#endif