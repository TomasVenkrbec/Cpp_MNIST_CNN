#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <memory>
#include "layer.hpp"
#include "dataset.hpp"

class Model {
private:
    Layer* input_layer = NULL;
    DatasetLoader dataset;
    // TODO: Add optimizer
    // TODO: Add loss

public:
    /**
     * @brief Add a Layer object to model
     * 
     * @param layer Layer to be added into model
     */
    void add_layer(Layer* layer);
    
    /**
     * @brief Prints the network to standard output
     */
    void print_model();
    
    void compile(); // Dataset, optimizer, loss, callbacks
};

#endif