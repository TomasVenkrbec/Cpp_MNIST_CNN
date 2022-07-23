#include <iostream>
#include "matrix.hpp"
#include "dataset.hpp"
#include "layer.hpp"
#include "layers/denselayer.hpp"
#include "layers/convlayer.hpp"
#include "model.hpp"

using namespace std;

int main() {
    // Initialize network
    Model model;
    model.add_layer(new ConvLayer(3, 4)); // 28x28 - padding
    model.add_layer(new ConvLayer(3, 8)); // 28x28 - padding
    // Downsample to 14x14
    model.add_layer(new ConvLayer(3, 16)); // 12x12 - no padding
    model.add_layer(new ConvLayer(3, 32)); // 12x12 - padding
    // Downsample to 6x6
    model.add_layer(new ConvLayer(3, 32)); // 6x6 - padding
    model.add_layer(new DenseLayer(128));
    model.add_layer(new DenseLayer(32));
    model.add_layer(new DenseLayer(10));
    
    // Print network
    model.print_model();

    // Load data
    DatasetLoader *dataset = get_dataset_loader("mnist");

    return 0;
}