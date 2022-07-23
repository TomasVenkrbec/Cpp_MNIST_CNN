#include <iostream>
#include "matrix.hpp"
#include "dataset.hpp"
#include "layers/denselayer.hpp"
#include "layers/convlayer.hpp"
#include "layers/avgpoollayer.hpp"
#include "losses/categoricalcrossentropy.hpp"
#include "optimizers/adam.hpp"
#include "model.hpp"

using namespace std;

int main() {
    // Initialize network
    Model model;
    model.add_layer(new ConvLayer(3, 4)); // 28x28 - padding
    model.add_layer(new ConvLayer(3, 8)); // 28x28 - padding
    model.add_layer(new AvgPoolLayer(2)); // Downsample to 14x14
    model.add_layer(new ConvLayer(3, 16)); // 12x12 - no padding
    model.add_layer(new ConvLayer(3, 32)); // 12x12 - padding
    model.add_layer(new AvgPoolLayer(2)); // Downsample to 6x6
    model.add_layer(new ConvLayer(3, 32)); // 6x6 - padding
    model.add_layer(new DenseLayer(128));
    model.add_layer(new DenseLayer(32));
    model.add_layer(new DenseLayer(10));
    
    // Load data
    DatasetLoader *dataset = get_dataset_loader("mnist");

    // Initialize loss function
    CategoricalCrossentropy *loss = new CategoricalCrossentropy();

    // Initialize optimizer
    float learning_rate = 0.001;
    Adam *optimizer = new Adam(learning_rate);

    // Compile the model
    model.compile(dataset, loss, optimizer);

    // Print network
    model.print_model();

    return 0;
}