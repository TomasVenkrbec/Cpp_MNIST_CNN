#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "../aliases.hpp"
#include "../layer.hpp"
#include "../matrix.hpp"

class Flatten: public Layer {
private:
    unsigned int kernel_size;

public:
    /**
     * @brief Construct a Flatten object with selected kernel size
     */
    Flatten();
    
    /**
     * @brief Process one sample of input data
     * 
     * @param sample Sample of input data
     * @return Output of layer operation on input sample
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