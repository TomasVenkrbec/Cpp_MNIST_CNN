#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <vector>
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

    /**
     * @brief Calculate and add derivatives of activation function (assuming the rest of derivative chain is already calculated)
     * 
     * @param sample_idx Index of sample from batch
     */
    void add_activation_derivatives(unsigned int sample_idx);

    /**
     * @brief Calculate and add derivatives of biases
     */
    void add_bias_derivatives();

    /**
     * @brief Calculate and add derivatives of weights
     * 
     * @param sample_idx Index of sample from batch
     */
    void add_weight_derivatives(unsigned int sample_idx);

    /**
     * @brief Calculate and add derivatives of all neuron activations from previous layer 
     */
    void add_prev_layer_derivatives();

    /**
     * @brief Get the derivatives for kernel with given index
     * 
     * @param kernel_idx Index of kernel
     * @return Matrix of derivatives corresponding to given kernel
     */
    Matrix* get_derivatives_for_kernel(unsigned int kernel_idx);

    /**
     * @brief Get the neurons for kernel with given index
     * 
     * @param kernel_idx Index of kernel
     * @return Vector of neurons corresponding to given kernel
     */
    std::vector<Neuron*> get_neurons_for_kernel(unsigned int kernel_idx);
};

#endif