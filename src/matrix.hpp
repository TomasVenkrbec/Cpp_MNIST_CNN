#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {
private:
    unsigned int x_size, y_size;
    std::vector<std::vector<float>> data;

public:
    /**
     * @brief Matrix object constructor
     * 
     * @param x_size Number of rows in matrix
     * @param y_size Number of cols in matrix
     */
    Matrix(unsigned int x_size, unsigned int y_size);

    /**
     * @brief Matrix object destructor
     */
    ~Matrix();
    
    /**
     * @brief Get the x size of matrix
     * 
     * @return x size of matrix
     */
    unsigned int get_x_size();

    /**
     * @brief Get the y size of matrix
     * 
     * @return y size of matrix
     */
    unsigned int get_y_size();

    /**
     * @brief Get value at given coordinates
     * 
     * @param x x coordinate
     * @param y y coordinate
     * @return Value on given coordinates
     */
    float at(unsigned int x, unsigned int y);

    /**
     * @brief Prints the matrix to console
     */
    void print_matrix();

    /**
     * @brief Perform one step of convolution on matrix with selected kernel on given coordinates
     * 
     * @param kernel Convolutional kernel
     * @param start_x x coordinate where start of kernel matrix will be (negative coordinates are enabled due to padding, but not counted)
     * @param start_y y coordinate where start of kernel matrix will be (negative coordinates are enabled due to padding, but not counted)
     * @return Resulting value
     */
    float convolve(Matrix* kernel, int start_x, int start_y);

    /**
     * @brief Calculate average value of matrix over area covered by kernel
     * 
     * @param kernel Averaging kernel
     * @param start_x x coordinate where start of kernel matrix will be 
     * @param start_y y coordinate where start of kernel matrix will be
     * @return Resulting average value
     */
    float get_avg(Matrix* kernel, int start_x, int start_y);

    /**
     * @brief Set given coordinate of the matrix to selected value
     * 
     * @param x Selected row of matrix
     * @param y Selected column of matrix
     * @param value Value to be set on given coordinate
     */
    void set_matrix(unsigned int x, unsigned int y, float value);

    /**
     * @brief Initialize the entire matrix using values from vector
     * 
     * @param vector Vector from which the new matrix will be initialized
     */
    void set_matrix_from_vector(std::vector<float> vector);

    /**
     * @brief Overload + operator to enable matrix + matrix operation
     * 
     * @return Matrix addition result 
     */
    Matrix operator+(const Matrix&);

    /**
     * @brief Overload + operator to enable matrix + scalar operation
     * 
     * @return Matrix addition result 
     */
    Matrix operator+(const float&);

    /**
     * @brief Overload - operator to enable matrix - matrix operation
     * 
     * @return Matrix subtract result 
     */
    Matrix operator-(const Matrix&);

    /**
     * @brief Overload - operator to enable matrix - scalar operation
     * 
     * @return Matrix subtract result 
     */
    Matrix operator-(const float&);

    /**
     * @brief Overload * operator to enable matrix * matrix operation
     * 
     * @return Matrix multiplication result 
     */
    Matrix operator*(const Matrix&);

    /**
     * @brief Overload * operator to enable matrix * scalar operation
     * 
     * @return Matrix multiplication result 
     */
    Matrix operator*(const float&);
};

#endif