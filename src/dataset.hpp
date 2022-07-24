#ifndef DATASET_H
#define DATASET_H

#include "matrix.hpp"
#include <string>
#include <vector>

class DataSample {
private:
    unsigned int label;
    std::vector<Matrix*> data; // Vector of matrices (channels)

public:
    /**
     * @brief DataSample object constructor
     * @param label Label of the sample
     * @param data Matrix containing the data of the sample
     */
    DataSample(unsigned int label, std::vector <Matrix*> data);

    /**
     * @brief DataSample object destructor
     */
    ~DataSample();

    /**
     * @brief Get image data from sample
     * 
     * @return Vector of matrices (channels)
     */
    std::vector<Matrix*> get_data();

    /**
     * @brief Get label value from sample
     * 
     * @return Label value
     */
    unsigned int get_label();
};

class DatasetLoader {
private:
    std::vector<DataSample*> train_samples;
    std::vector<DataSample*> val_samples;
    unsigned int resolution;
    unsigned int max_label;
    unsigned int channels;
    unsigned int current_sample_train = 0; // Index of current train sample
    unsigned int current_sample_val = 0; // Index of current val sample 

public:
    unsigned int batch_size;

    /**
     * @brief Get batch of training data
     * 
     * @return Vector of training data samples
     */
    std::vector<DataSample*> get_train_batch();

    /**
     * @brief Reset batch generator of train data
     */
    void reset_train_batch_generator();

    /**
     * @brief Get batch of validation data
     * 
     * @return Vector of validation data samples
     */
    std::vector<DataSample*> get_val_batch();
  
    /**
     * @brief Reset batch generator of validation data
     */
    void reset_val_batch_generator();

    /**
     * @brief Get resolution of samples
     * 
     * @return Resolution of samples
     */
    unsigned int get_resolution();

    /**
     * @brief Get number of channels in samples
     * 
     * @return Number of channels
     */
    unsigned int get_channels();

    /**
     * @brief Get the max value of label
     * 
     * @return Maximal value of label
     */
    unsigned int get_max_label();

    /**
     * @brief Get the max value of label
     * 
     * @return Maximal value of label
     */
    unsigned int get_train_sample_count();

    /**
     * @brief Get the max value of label
     * 
     * @return Maximal value of label
     */
    unsigned int get_val_sample_count();

    /**
     * @brief Add a new sample to train dataset
     * 
     * @param sample Added sample 
     */
    void add_train_sample(DataSample* sample);

    /**
     * @brief Add a new sample to val dataset
     * 
     * @param sample Added sample 
     */
    void add_val_sample(DataSample* sample);

    /**
     * @brief Load MNIST dataset into DatasetLoader
     */
    void load_mnist_dataset();

    /**
     * @brief Load MNIST csv file
     * 
     * @param name Name of MNIST subset {train, test}
     */
    void load_mnist_file(std::string name);
};


/**
 * @brief Check if line from dataset in CSV format has correct label and correct resolution
 *
 * @param line Line from CSV dataset to be checked
 * @param max_label Max value of label
 * @param x_size Number of rows in image data
 * @param y_size Number of cols in image data
 * @return DataSample object if the data is correct, throws error otherwise
 */
DataSample* parse_csv_line(std::string line, unsigned int max_label, unsigned int x_size, unsigned int y_size, unsigned int channels);

/**
 * @brief Get the dataset loader object for selected dataset
 * 
 * @param dataset Name of selected dataset (currently supported - {"mnist"})
 * @return Dataset loader object
 */
DatasetLoader* get_dataset_loader(std::string dataset);

#endif