#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "dataset.hpp"
#include "matrix.hpp"

using namespace std;

#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE 10000
#define MNIST_RESOLUTION 28
#define MNIST_MAX_LABEL 9

void DatasetLoader::add_train_sample(DataSample* sample) {
    this->train_samples.push_back(*sample);
}

void DatasetLoader::add_test_sample(DataSample* sample) {
    this->test_samples.push_back(*sample);
}

DataSample::DataSample(unsigned int label, Matrix* data) {
    this->label = label;
    this->data = data;
}

DataSample::~DataSample() {

}

DataSample* parse_csv_line(string line, unsigned int max_label, unsigned int x_size, unsigned int y_size) {
    vector<float> data_vector;
    unsigned int label;

    stringstream line_ss;
    line_ss.str(line);
    string num;
    
    // Get label from line
    getline(line_ss, num, ',');
    label = stoi(num);
    if (label < 0 || label > max_label) {
        cerr << "ERROR: Label contains invalid value: " << label << endl;
        throw label;
    }

    // Get data from line
    while(getline(line_ss, num, ',')) {
        data_vector.push_back(stod(num));
    }
    
    // Initialize data matrix
    Matrix *data_matrix = new Matrix(x_size, y_size);
    data_matrix->set_matrix_from_vector(data_vector);

    // Create sample object and return it
    return new DataSample(label, data_matrix);
}

void load_mnist_file(DatasetLoader* dataset, string name) {
    string line;
    ifstream file;
    file.open("mnist_csv/mnist_" + name + ".csv");

    // Check if file exists 
    if (!file.is_open()) {
        cerr << "ERROR: Failed to open .csv file for " + name + " subset of MNIST!" << endl;
        throw;
    }
    
    unsigned int sample_count = 0;
    while (getline(file, line)) { // Parse entire file line by line
        sample_count++;

        try {
            // Parse line
            DataSample* sample = parse_csv_line(line, MNIST_MAX_LABEL, MNIST_RESOLUTION, MNIST_RESOLUTION);

            // Add sample to dataset
            if (name == "train") {
                dataset->add_train_sample(sample);
            }
            else {
                dataset->add_test_sample(sample);
            }
        }
        catch (...) {
            cerr << "ERROR: Encountered bad data in " + name + " subset of MNIST dataset, on line: " << sample_count << endl;
            throw;
        }
    }
    cout << "MNIST " + name + " sample count: " << sample_count << endl;
    file.close();
}

DatasetLoader* get_mnist_dataset() {
    DatasetLoader* dataset = new DatasetLoader;
    load_mnist_file(dataset, "train"); // Load training data
    load_mnist_file(dataset, "test"); // Load test data

    return dataset;
}

DatasetLoader* get_dataset_loader(string dataset)
{
    if (dataset == "mnist") {
        cout << "Loading MNIST dataset." << endl;

        return get_mnist_dataset();
    }
    else {
        cerr << "ERROR: Dataset " << dataset << " is not available!" << endl;
        throw; 
    }
}