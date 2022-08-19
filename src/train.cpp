#include <iostream>
#include <vector>
#include "matrix.hpp"
#include "dataset.hpp"
#include "layers/dense.hpp"
#include "layers/conv.hpp"
#include "layers/avgpool.hpp"
#include "layers/flatten.hpp"
#include "layers/softmax.hpp"
#include "losses/categoricalcrossentropy.hpp"
#include "optimizers/sgd.hpp"
#include "regularizers/l2.hpp"
#include "activations/relu.hpp"
#include "activations/sigmoid.hpp"
#include "callback.hpp"
#include "callbacks/accuracy.hpp"
#include "model.hpp"

using namespace std;

int main() {
    // CURRENT LIMITATIONS OF NETWORKS
    // Fully connected networks need to start with Flatten layer -- maybe add Input layer?

    // Initialize network
    Model model;
    /*model.add_layer(new Conv2D(3, 4, new ReLU())); // 28x28 - padding
    model.add_layer(new Conv2D(3, 8, new ReLU())); // 28x28 - padding
    model.add_layer(new AvgPool(2)); // Downsample to 14x14
    model.add_layer(new Conv2D(3, 16, new ReLU(), false)); // 12x12 - no padding
    model.add_layer(new Conv2D(3, 32, new ReLU())); // 12x12 - padding
    model.add_layer(new AvgPool(2)); // Downsample to 6x6
    model.add_layer(new Conv2D(3, 32, new ReLU())); // 6x6 - padding
    model.add_layer(new Flatten());
    model.add_layer(new Dense(128, new ReLU()));
    model.add_layer(new Dense(32, new ReLU()));
    model.add_layer(new Dense(10, new Sigmoid()));
    model.add_layer(new Softmax());*/

    model.add_layer(new Flatten());
    model.add_layer(new Dense(64, new ReLU()));
    model.add_layer(new Dense(64, new ReLU()));
    model.add_layer(new Dense(10, new Sigmoid()));
    model.add_layer(new Softmax());
    
    // Load data
    DatasetLoader* dataset = get_dataset_loader("mnist");
    dataset->batch_size = 32;

    // Initialize loss function
    unsigned int moving_average_samples = 100;
    Loss* loss = new CategoricalCrossentropy(moving_average_samples);

    // Initialize optimizer
    float learning_rate = 0.01;
    Optimizer* optimizer = new SGD(learning_rate);

    // Initialize regularizer
    float reg_factor_weights = 1e-4;
    float reg_factor_bias = 1e-4;
    Regularizer* regularizer = new L2(reg_factor_weights, reg_factor_bias);

    // Initialize callbacks
    vector<Callback*> callback_vector;
    callback_vector.push_back(new Accuracy(moving_average_samples));

    // Compile the model
    model.compile(dataset, loss, optimizer, regularizer, callback_vector);

    // Print network
    model.print_model();

    // Train network
    unsigned int max_epochs = 100;
    model.fit(max_epochs);

    return 0;
}