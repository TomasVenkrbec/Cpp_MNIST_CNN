#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>
#include "dataset.hpp"
#include "matrix.hpp"

using namespace std;

#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE 10000
#define MNIST_RESOLUTION 28
#define MNIST_CHANNELS 1
#define MNIST_MAX_LABEL 9

void DatasetLoader::add_train_sample(DataSample* sample) {
    this->train_samples.push_back(sample);
}

void DatasetLoader::add_val_sample(DataSample* sample) {
    this->val_samples.push_back(sample);
}

unsigned int DatasetLoader::get_resolution() {
    return this->resolution;
}

unsigned int DatasetLoader::get_max_label() {
    return this->max_label;
}

unsigned int DatasetLoader::get_channels() {
    return this->channels;
}

unsigned int DatasetLoader::get_train_sample_count() {
    return this->train_samples.size();
}

unsigned int DatasetLoader::get_val_sample_count() {
    return this->val_samples.size();
}

vector<DataSample*> DatasetLoader::get_train_batch() {
    if (this->current_sample_train + this->batch_size > this->train_samples.size()) {
        cerr << "ERROR: Not enough training samples left to complete the batch" << endl;
        throw; 
    }

    vector<DataSample*> batch_data;
    while (batch_data.size() < this->batch_size) {
        batch_data.push_back(this->train_samples[this->current_sample_train++]);
    }

    return batch_data;
}

void DatasetLoader::reset_train_batch_generator() {
    this->current_sample_train = 0;

    random_device rd;
    mt19937 gen(rd()); // Randomizer
    shuffle(begin(this->train_samples), end(train_samples), gen); // Shuffle training samples
}

vector<DataSample*> DatasetLoader::get_val_batch() {
    if (this->current_sample_val + this->batch_size > this->val_samples.size()) {
        cerr << "ERROR: Not enough validation samples left to complete the batch" << endl;
        throw; 
    }

    vector<DataSample*> batch_data;
    while (batch_data.size() < this->batch_size) {
        batch_data.push_back(this->val_samples[this->current_sample_val++]);
    }

    return batch_data;
}

void DatasetLoader::reset_val_batch_generator() {
    this->current_sample_val = 0;
}

DataSample::DataSample(unsigned int label, vector<Matrix*> data) {
    this->label = label;
    this->data = data;
}

DataSample::~DataSample() {

}

DataSample* parse_csv_line(string line, unsigned int max_label, unsigned int x_size, unsigned int y_size, unsigned int channels) {
    vector<vector<float>> data_channels;
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
    while (getline(line_ss, num, ',')) {
        data_vector.push_back(stof(num));
        if (data_vector.size() == x_size * y_size) { // Gathered complete channel
            data_channels.push_back(data_vector);
            data_vector.clear();
        }
    }
    
    // Check if number of channels loaded is correct
    if (data_channels.size() != channels) {
        cerr << "ERROR: The number of channels in data is incorrect." << endl;
        throw data_channels.size();
    }

    // Initialize data matrix
    vector<Matrix*> data_matrix_vector;
    for (unsigned i = 0; i < channels; i++) {
        Matrix* data_matrix = new Matrix(x_size, y_size);
        data_matrix->set_matrix_from_vector(data_channels[i]); // Correct resolution of data is checked within this function
        data_matrix_vector.push_back(data_matrix);
    }

    // Create sample object and return it
    return new DataSample(label, data_matrix_vector);
}

void DatasetLoader::load_mnist_file(string name) {
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
            DataSample* sample = parse_csv_line(line, MNIST_MAX_LABEL, MNIST_RESOLUTION, MNIST_RESOLUTION, MNIST_CHANNELS);

            // Add sample to dataset
            if (name == "train") {
                this->add_train_sample(sample);
            }
            else {
                this->add_val_sample(sample);
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

void DatasetLoader::load_mnist_dataset() {
    this->resolution = MNIST_RESOLUTION;
    this->max_label = MNIST_MAX_LABEL;
    this->channels = MNIST_CHANNELS;
    this->load_mnist_file("train"); // Load training data
    this->load_mnist_file("test"); // Load val data
}

DatasetLoader* get_dataset_loader(string dataset)
{
    if (dataset == "mnist") {
        cout << "Loading MNIST dataset." << endl;

        DatasetLoader* dataset = new DatasetLoader;
        dataset->load_mnist_dataset();
        return dataset;
    }
    else {
        cerr << "ERROR: Dataset " << dataset << " is not available!" << endl;
        throw; 
    }
}

std::vector<Matrix*> DataSample::get_data() {
    return this->data;
}

unsigned int DataSample::get_label() {
    return this->label;
}
