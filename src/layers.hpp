#ifndef LAYERS_H
#define LAYERS_H

#include "neuron.hpp"
#include <string>
#include <vector>

class Layer {
protected:
    std::string name;
    std::vector<Neuron> neurons;
    Layer* prev_layer = NULL;
    Layer* next_layer = NULL;

public:
    /**
     * @brief Layer object constructor
     * 
     * @param neuron_count Layer neuron count
     */
    Layer(unsigned int neuron_count);

    /**
     * @brief Layer object destructor
     */
    ~Layer();

    /**
     * @brief Get the count of neurons in layer
     * 
     * @return Layer neuron count
     */
    unsigned int get_neuron_count();
};

class DenseLayer: public Layer {
public:
    using Layer::Layer;
    std::string name = "DenseLayer";
};

class ConvLayer: public Layer {
private:
    unsigned int kernel_size;
    unsigned int kernel_count;

public:
    std::string name = "ConvLayer";

    /**
     * @brief Construct a ConvLayer object with selected kernel size and count
     * 
     * @param kernel_size Size of kernel (kernel_size * kernel_size)
     * @param kernel_count Kernel (filter) count in layer
     * @return Initialized ConvLayer object
     */
    ConvLayer(unsigned int kernel_size, unsigned int kernel_count);
};

#endif