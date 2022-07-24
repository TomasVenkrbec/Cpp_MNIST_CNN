#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <vector>
#include "../layer.hpp"
#include "../matrix.hpp"

class Softmax : public Layer {
public:
    /**
     * @brief Softmax layer constructor
     */
    Softmax();

    /**
     * @brief Process one sample of input data
     * 
     * @param sample Sample of input data
     * @return Output of layer operation on input sample
     */
    std::vector<Matrix*> process_sample(std::vector<Matrix*> sample);

    /**
     * @brief Calculate output shape of layer
     * 
     * @param input_shape Input shape ([x,y,channels])
     */
    void calculate_output_shape(unsigned int input_shape[3]); 
};

#endif