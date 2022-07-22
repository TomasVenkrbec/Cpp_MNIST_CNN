#include <iostream>
#include "matrix.hpp"
#include "dataset.hpp"

using namespace std;

int main() {
    DatasetLoader *dataset = get_dataset_loader("mnist");
    return 0;
}