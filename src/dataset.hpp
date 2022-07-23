#ifndef DATASET_H
#define DATASET_H

#include "matrix.hpp"
#include <string>
#include <vector>

class DataSample {
private:
    unsigned int label;
    Matrix *data;

public:
    /**
     * @brief DataSample object constructor
     * @param label Label of the sample
     * @param data Matrix containing the data of the sample
     */
    DataSample(unsigned int label, Matrix* data);

    /**
     * @brief DataSample object destructor
     */
    ~DataSample();
};

class DatasetLoader {
private:
    std::vector<DataSample> train_samples;
    std::vector<DataSample> test_samples;

public:
    /**
     * @brief Add a new sample to train dataset
     * 
     * @param sample Added sample 
     */
    void add_train_sample(DataSample* sample);

    /**
     * @brief Add a new sample to test dataset
     * 
     * @param sample Added sample 
     */
    void add_test_sample(DataSample* sample);
};

/**
 * @brief Get the DatasetLoader object for MNIST dataset
 * 
 * @return DatasetLoader object for MNIST dataset
 */
DatasetLoader* get_mnist_dataset();

/**
 * @brief Load MNIST csv file
 * 
 * @param dataset Object for loaded data
 * @param name Name of MNIST subset {train, test}
 */
void load_mnist_file(DatasetLoader* dataset, std::string name);

/**
 * @brief Check if line from dataset in CSV format has correct label and correct resolution
 *
 * @param line Line from CSV dataset to be checked
 * @param max_label Max value of label
 * @param x_size Number of rows in image data
 * @param y_size Number of cols in image data
 * @return DataSample object if the data is correct, throws error otherwise
 */
DataSample* parse_csv_line(std::string line, unsigned int max_label, unsigned int x_size, unsigned int y_size);

/**
 * @brief Get the dataset loader object for selected dataset
 * 
 * @param dataset Name of selected dataset (currently supported - {"mnist"})
 * @return Dataset loader object
 */
DatasetLoader* get_dataset_loader(std::string dataset);

#endif