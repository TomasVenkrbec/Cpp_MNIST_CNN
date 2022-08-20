#ifndef INPUT_H
#define INPUT_H

#include "../layer.hpp"

class Input: public Layer {
private:
    unsigned int kernel_size;

public:
    /**
     * @brief Construct an Input layer
     * 
     */
    Input();

    /**
     * @brief Calculate output shape of layer
     * 
     * @param input_shape Input shape ([x,y,channels])
     */
    void calculate_output_shape(unsigned int input_shape[3]); 

    /**
     * @brief Pass data to first layer
     * 
     * @param data Data to be passed
     * @return Unchanged data
     */
    Matrix* process_channel(Matrix* data);

    /**
     * @brief Initialize neurons of layer
     */
    void initialize_neurons();
};

#endif