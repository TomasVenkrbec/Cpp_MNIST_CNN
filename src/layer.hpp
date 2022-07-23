#ifndef LAYER_H
#define LAYER_H

#include "neuron.hpp"
#include <string>
#include <vector>

class Layer {
protected:
    std::vector<Neuron> neurons;
    Layer* prev_layer = NULL;
    Layer* next_layer = NULL;

public:
    std::string name;

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

    /**
     * @brief Add the pointer to next layer
     * 
     * @param layer Next layer
     */
    void add_next_layer(Layer* layer);

    /**
     * @brief Add the pointer to previous layer
     * 
     * @param layer Previous layer
     */
    void add_prev_layer(Layer* layer);

    /**
     * @brief Get the pointer to previous layer
     * 
     * @param layer Previous layer pointer
     */
    Layer* get_prev_layer();

    /**
     * @brief Get the pointer to next layer
     * 
     * @param layer Next layer pointer
     */
    Layer* get_next_layer();

    /**
     * @brief Perform forward pass
     */
    virtual void forward();
};

#endif