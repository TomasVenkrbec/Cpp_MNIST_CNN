#include <iostream>
#include <vector>
#include "matrix.hpp"

using namespace std;

Matrix::Matrix(unsigned int x_size, unsigned int y_size) {
    this->x_size = x_size;
    this->y_size = y_size;
    for (unsigned int i = 0; i < this->x_size; i++) {

        vector<float> row;
        for (unsigned int j = 0 ; j < this->y_size; j++) {
            row.push_back(0.0); // Default value
        }
        this->data.push_back(row);
    }
}

Matrix::~Matrix() {
    for (unsigned int i = 0; i < this->x_size; i++) {
        this->data[i].clear();
    }
    this->data.clear();
}

unsigned int Matrix::get_x_size() {
    return this->x_size;
}

unsigned int Matrix::get_y_size() {
    return this->y_size;
}

float Matrix::at(unsigned int x, unsigned int y) {
    if (x > this->x_size || y > this->y_size) {
        cerr << "ERROR: Attempting to access out-of-bounds data (" << x << "," << y << ") from matrix (" << this->x_size << "," << this->y_size << ")." << endl;
        throw;
    }
    return this->data[x][y];
}

void Matrix::print_matrix() {
    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            cout << this->data[i][j] << " ";
        }
        cout << endl;
    }
}

void Matrix::set_matrix(unsigned int x, unsigned int y, float value) {
    if (x >= this->x_size) {
        cerr << "ERROR: Matrix x coordinate (" << x << ") is out of bounds for matrix with row count: " << this->x_size << endl;
        throw;
    }

    if (y >= this->y_size) {
        cerr << "ERROR: Matrix y coordinate (" << y << ") is out of bounds for matrix with column count: " << this->y_size << endl;
        throw;
    }

    this->data[x][y] = value;
}

void Matrix::set_matrix_from_vector(std::vector<float> vector){
    if (vector.size() != this->x_size * this->y_size) {
        cerr << "ERROR: Size mismatch, matrix with dimensions (" << this->x_size << "," << this->y_size << ") can't be set using vector of size: " << vector.size() << "." << endl; 
        throw;
    }
    
    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            this->data[i][j] = vector[i * this->x_size + j]; 
        }
    }
}

float Matrix::convolve(Matrix* kernel, int start_x, int start_y) {
    float result = 0.0;

    for (unsigned int i = 0; i < kernel->get_x_size(); i++) { // Iterate over x coordinates of kernel
        for (unsigned int j = 0; j < kernel->get_y_size(); j++) { // Iterate over y coordinates of kernel
            // Out-of-bounds coordinates are not an error due to padding, but they're simply ignored
            if (i + start_x < 0 || i + start_x >= this->get_x_size() || j + start_y < 0 || j + start_y >= this->get_y_size()) {
                continue;
            }

            // Calculate kernel * data on valid position, add to result
            result += kernel->at(i, j) * this->data[i + start_x][j + start_y];
        }
    }

    return result;
}

float Matrix::get_avg(Matrix* kernel, int start_x, int start_y) {
    float result = 0.0;

    for (unsigned int i = 0; i < kernel->get_x_size(); i++) { // Iterate over x coordinates of kernel
        for (unsigned int j = 0; j < kernel->get_y_size(); j++) { // Iterate over y coordinates of kernel
            // Out-of-bounds coordinates are an error, there can't be padding
            if (i + start_x < 0 || i + start_x >= this->get_x_size() || j + start_y < 0 || j + start_y >= this->get_y_size()) {
                cerr << "ERROR: Coordinate (" << i + start_x << "," << j + start_y << ") is out of bounds for matrix of size (" << this->get_x_size() << "," << this->get_y_size() << ")" << endl;
                throw;
            }
            
            // Sum the values first
            result += this->data[i + start_x][j + start_y];
        }
    }

    // Calculate average
    return result / (kernel->get_x_size() * kernel->get_y_size());
}

Matrix Matrix::operator+(const float& value) {
    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] + value; 
        }
    }
    return res;
}

Matrix Matrix::operator+(const Matrix& matrix) {
    if (this->x_size != matrix.x_size || this->y_size != matrix.y_size) {
        cerr << "ERROR: Can't add matrices with different dimensions. Dimensions: (" << this->x_size << "," << this->y_size << "), (" << matrix.x_size << "," << matrix.y_size << ")."  << endl;
        throw;
    }

    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] + matrix.data[i][j]; 
        }
    }
    return res;
}

Matrix Matrix::operator-(const float& value) {
    return *this + (-value);
}

Matrix Matrix::operator-(const Matrix& matrix) {
    if (this->x_size != matrix.x_size || this->y_size != matrix.y_size) {
        cerr << "ERROR: Can't subtract matrices with different dimensions. Dimensions: (" << this->x_size << "," << this->y_size << "), (" << matrix.x_size << "," << matrix.y_size << ")."  << endl;
        throw;
    }

    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] - matrix.data[i][j]; 
        }
    }
    return res;
}

Matrix Matrix::operator*(const float& value) {
    Matrix res = Matrix(this->x_size, this->y_size);

    for (unsigned int i = 0; i < this->x_size; i++) { // Rows
        for (unsigned int j = 0; j < this->y_size; j++) { // Columns
            res.data[i][j] = this->data[i][j] * value; 
        }
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& matrix) {
    // (x,y) * (y,z) == (x,z)
    if (this->y_size != matrix.x_size) {
        cerr << "ERROR: Matrix multiplication size mismatch. Dimensions 1 of first matrix and dimension 0 of second matrix need to match. Dimensions: (" << this->x_size << "," << this->y_size << "), (" << matrix.x_size << "," << matrix.y_size << ")."  << endl;
        throw;
    }

    Matrix res = Matrix(this->x_size, matrix.y_size);

    for (unsigned int i = 0; i < res.x_size; i++) { // Rows of final matrix
        for (unsigned int j = 0; j < res.y_size; j++) { // Columns of final matrix
            float sum = 0.0;
            for (unsigned int k = 0; k < this->y_size; k++) { // Across the same-sized dimension
                sum += this->data[i][k] * matrix.data[k][j];
            }
            res.data[i][j] = sum;
        }
    }    

    return res;
}