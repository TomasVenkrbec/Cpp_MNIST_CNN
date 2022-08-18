#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "aliases.hpp"
#include "layer.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include "callback.hpp"
#include "matrix.hpp"

class Model {
private:
    Layer* input_layer = NULL;
    Layer* output_layer = NULL;
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
     * @brief Perform single step (training or validation)
     * 
     * @param batch_data Batch of data
     * @param is_training Whether or not the training will happen, or only validation
     */
    void step(std::vector<DataSample*> batch_data, bool is_training);

    /**
     * @brief Train model for 1 epoch
     */
    void train();

    /**
     * @brief Perform validation of model
     */
    void validate();
    
    /**
     * @brief Reset all callbacks
     */
    void reset_callbacks();

    /**
     * @brief Perform forwards pass of data
     * 
     * @param data Batch of input data
     * @param labels Batch of input labels
     * @return Batch of feature maps, model output given input data
     */
    Batch forward_pass(Batch data);

    /**
     * @brief Perform backpropagation of model
     * @param y_pred Batch of predicted labels
     * @param y_true Batch of ground truth labels 
     */
    void backprop(Batch y_pred, Batch y_true);

    /**
     * @brief Update all weights of a model after backpropagation
     */
    void update_weights();
};

#endif