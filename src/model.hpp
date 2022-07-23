#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <memory>
#include "layer.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"
#include "loss.hpp"

class Model {
private:
    Layer* input_layer = NULL;
    DatasetLoader* dataset;
    Optimizer* optimizer;
    Loss* loss;

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
    
    void compile(DatasetLoader* dataset, Loss* loss, Optimizer* optimizer); // TODO: Add callbacks
};

#endif