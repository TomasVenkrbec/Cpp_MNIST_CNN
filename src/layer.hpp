#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>
#include "aliases.hpp"
#include "neuron.hpp"
#include "activation.hpp"
#include "matrix.hpp"
#include "initializer.hpp"

class Layer {
protected:
    std::vector<Neuron*> neurons;
    Activation* activation = NULL;
    Initializer* initializer = NULL;
    Layer* prev_layer = NULL;
    Layer* next_layer = NULL;
    unsigned int output_shape[3];
    unsigned int trainable_weights_count = 0;
    bool process_by_channel = true; // true - process image channel by channel, false - process image sample by sample

public:
    unsigned int batch_size;
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
     * @brief Get the count of trainable weights in layer
     * 
     * @return Trainable weights count
     */
    unsigned int get_trainable_weights_count();

    /**
     * @brief Get output shape of layer
     * 
     * @return Output shape
     */
    unsigned int* get_output_shape();

    /**
     * @brief Get the activation function of layer
     * 
     * @return Activation function
     */
    Activation* get_activation();

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
     * @brief Get neurons from layer
     * 
     * @return Vector of neurons from layer 
     */
    std::vector<Neuron*> get_neurons();

    /**
     * @brief Forward propagation function
     * 
     * @param data Vector of vectors of channels - batch of feature maps
     * @return Batch of feature maps
     */
    Batch forward(Batch data);

    /**
     * @brief Process one channel of input data
     * 
     * @param channel Channel of input data
     * @return Output of layer operation on input channel
     */
    virtual Matrix* process_channel(Matrix* channel);

    /**
     * @brief Process one sample of input data
     * 
     * @param sample Sample of input data
     * @return Output of layer operation on input sample
     */
    virtual Sample process_sample(Sample sample);

    /**
     * @brief Calculate output shape of layer
     * 
     * @param input_shape Input shape of layer ([x,y,channels])
     */
    virtual void calculate_output_shape(unsigned int input_shape[3]);

    /**
     * @brief Initialize neurons of layer
     */
    virtual void initialize_neurons();

    /**
     * @brief Get derivative of activation function given activation value
     * 
     * @param activation Value of activation
     * @return Activation derivative
     */
    virtual float get_activation_derivative(float activation);

    /**
     * @brief Calculate and add derivatives of activation function (assuming the rest of derivative chain is already calculated)
     * 
     * @param sample_idx Index of sample from batch
     */
    virtual void add_activation_derivatives(unsigned int sample_idx);

    /**
     * @brief Calculate and add derivatives of biases
     */
    virtual void add_bias_derivatives();

    /**
     * @brief Calculate and add derivatives of weights
     * 
     * @param sample_idx Index of sample from batch
     */
    virtual void add_weight_derivatives(unsigned int sample_idx);

    /**
     * @brief Calculate and add derivatives of all neuron activations from previous layer 
     */
    virtual void add_prev_layer_derivatives();

    /**
     * @brief Clear derivative of layer, in preparation for next sample calculation
     */
    virtual void clear_layer_derivatives();

};

#endif