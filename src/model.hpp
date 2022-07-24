#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <memory>
#include "layer.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include "callback.hpp"
#include "matrix.hpp"

class Model {
private:
    Layer* input_layer = NULL;
    unsigned int* input_shape;
    DatasetLoader* dataset;
    Optimizer* optimizer;
    Loss* loss;
    std::vector<Callback*> callbacks;

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

    /**
     * @brief Get the vector of callbacks
     * 
     * @return Vector of callbacks
     */
    std::vector<Callback*> get_callbacks();
    
    /**
     * @brief Validate and initialize the model
     * 
     * @param dataset DatasetLoader object for loaded dataset
     * @param loss Loss object
     * @param optimizer Optimizer object
     * @param callbacks Vector of callbacks
     */
    void compile(DatasetLoader* dataset, Loss* loss, Optimizer* optimizer, std::vector<Callback*> callbacks);

    /**
     * @brief Train the model for given number of epochs
     * 
     * @param max_epochs Maximum number of epochs
     */
    void fit(unsigned int max_epochs);

    /**
     * @brief Perform single training step
     */
    void step();

    /**
     * @brief Perform validation of model
     */
    void validate();

    /**
     * @brief Perform forwards pass of data
     * 
     * @param batch_data Vector of data samples (batch)
     * @return Batch of feature maps, model output given input data
     */
    std::vector<std::vector<Matrix*>> forward_pass(std::vector<DataSample*> batch_data);
};

#endif