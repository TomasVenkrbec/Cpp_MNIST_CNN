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
     * @brief Calculate and add derivatives of all neuron activations from previous layer 
     */
    void add_prev_layer_derivatives();
};

#endif