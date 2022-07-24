#include <iostream>
#include <vector>
#include "matrix.hpp"
#include "dataset.hpp"
#include "layers/dense.hpp"
#include "layers/conv.hpp"
#include "layers/avgpool.hpp"
#include "layers/flatten.hpp"
#include "losses/categoricalcrossentropy.hpp"
#include "optimizers/adam.hpp"
#include "activations/relu.hpp"
#include "activations/softmax.hpp"
#include "callback.hpp"
#include "callbacks/accuracy.hpp"
#include "model.hpp"

using namespace std;

int main() {
    // Initialize network
    Model model;
    model.add_layer(new Conv2D(3, 4, new ReLU())); // 28x28 - padding
    model.add_layer(new Conv2D(3, 8, new ReLU())); // 28x28 - padding
    model.add_layer(new AvgPool(2)); // Downsample to 14x14
    model.add_layer(new Conv2D(3, 16, new ReLU(), false)); // 12x12 - no padding
    model.add_layer(new Conv2D(3, 32, new ReLU())); // 12x12 - padding
    model.add_layer(new AvgPool(2)); // Downsample to 6x6
    model.add_layer(new Conv2D(3, 32, new ReLU())); // 6x6 - padding
    model.add_layer(new Flatten());
    model.add_layer(new Dense(128, new ReLU()));
    model.add_layer(new Dense(32, new ReLU()));
    model.add_layer(new Dense(10, new Softmax()));
    
    // Load data
    DatasetLoader* dataset = get_dataset_loader("mnist");
    dataset->batch_size = 2;

    // Initialize loss function
    CategoricalCrossentropy* loss = new CategoricalCrossentropy();

    // Initialize optimizer
    float learning_rate = 0.001;
    Adam* optimizer = new Adam(learning_rate);

    // Initialize callbacks
    vector<Callback*> callback_vector;
    callback_vector.push_back(new Accuracy());

    // Compile the model
    model.compile(dataset, loss, optimizer, callback_vector);

    // Print network
    model.print_model();

    // Train network
    unsigned int max_epochs = 3;
    model.fit(max_epochs);

    return 0;
}