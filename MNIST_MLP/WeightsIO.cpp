#include "WeightsIO.h"
#include <fstream>
#include <cstdint>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

bool WeightsIO::fileExists(const std::string& filename) {
    return access(filename.c_str(), F_OK) == 0;
}

bool WeightsIO::saveBinary(const std::string& filename,
    const Eigen::MatrixXd& W1,
    const Eigen::VectorXd& b1,
    const Eigen::MatrixXd& W2,
    const Eigen::VectorXd& b2) {
    if (W1.rows() != 256 || W1.cols() != 784) {
        throw std::runtime_error("W1 has invalid dimensions: " +
            std::to_string(W1.rows()) + "x" + std::to_string(W1.cols()) +
            ", expected 256x784");
    }

    if (b1.size() != 256) {
        throw std::runtime_error("b1 has invalid size: " +
            std::to_string(b1.size()) + ", expected 256");
    }

    if (W2.rows() != 10 || W2.cols() != 256) {
        throw std::runtime_error("W2 has invalid dimensions: " +
            std::to_string(W2.rows()) + "x" + std::to_string(W2.cols()) +
            ", expected 10x256");
    }

    if (b2.size() != 10) {
        throw std::runtime_error("b2 has invalid size: " +
            std::to_string(b2.size()) + ", expected 10");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    int64_t rows1 = W1.rows();
    int64_t cols1 = W1.cols();
    int64_t rows2 = W2.rows();
    int64_t cols2 = W2.cols();
    int64_t sizeB1 = b1.size();
    int64_t sizeB2 = b2.size();

    file.write(reinterpret_cast<const char*>(&rows1), sizeof(rows1));
    file.write(reinterpret_cast<const char*>(&cols1), sizeof(cols1));
    file.write(reinterpret_cast<const char*>(&rows2), sizeof(rows2));
    file.write(reinterpret_cast<const char*>(&cols2), sizeof(cols2));
    file.write(reinterpret_cast<const char*>(&sizeB1), sizeof(sizeB1));
    file.write(reinterpret_cast<const char*>(&sizeB2), sizeof(sizeB2));

    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to write dimensions to file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(W1.data()), W1.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to write W1 to file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(b1.data()), b1.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to write b1 to file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(W2.data()), W2.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to write W2 to file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(b2.data()), b2.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to write b2 to file: " + filename);
    }

    file.close();
    return true;
}

bool WeightsIO::loadBinary(const std::string& filename,
    Eigen::MatrixXd& W1,
    Eigen::VectorXd& b1,
    Eigen::MatrixXd& W2,
    Eigen::VectorXd& b2) {
    if (!fileExists(filename)) {
        throw std::runtime_error("Weights file not found: " + filename);
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    int64_t rows1, cols1, rows2, cols2, sizeB1, sizeB2;

    file.read(reinterpret_cast<char*>(&rows1), sizeof(rows1));
    file.read(reinterpret_cast<char*>(&cols1), sizeof(cols1));
    file.read(reinterpret_cast<char*>(&rows2), sizeof(rows2));
    file.read(reinterpret_cast<char*>(&cols2), sizeof(cols2));
    file.read(reinterpret_cast<char*>(&sizeB1), sizeof(sizeB1));
    file.read(reinterpret_cast<char*>(&sizeB2), sizeof(sizeB2));

    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to read dimensions from file: " + filename);
    }

    if (rows1 != 256 || cols1 != 784) {
        file.close();
        throw std::runtime_error("W1 dimension mismatch in file: got " +
            std::to_string(rows1) + "x" + std::to_string(cols1) +
            ", expected 256x784");
    }

    if (sizeB1 != 256) {
        file.close();
        throw std::runtime_error("b1 size mismatch in file: got " +
            std::to_string(sizeB1) + ", expected 256");
    }

    if (rows2 != 10 || cols2 != 256) {
        file.close();
        throw std::runtime_error("W2 dimension mismatch in file: got " +
            std::to_string(rows2) + "x" + std::to_string(cols2) +
            ", expected 10x256");
    }

    if (sizeB2 != 10) {
        file.close();
        throw std::runtime_error("b2 size mismatch in file: got " +
            std::to_string(sizeB2) + ", expected 10");
    }

    W1.resize(rows1, cols1);
    b1.resize(sizeB1);
    W2.resize(rows2, cols2);
    b2.resize(sizeB2);

    file.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to read W1 from file: " + filename);
    }

    file.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to read b1 from file: " + filename);
    }

    file.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to read W2 from file: " + filename);
    }

    file.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(double));
    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to read b2 from file: " + filename);
    }

    file.close();
    return true;
}

bool WeightsIO::saveText(const std::string& filename,
    const Eigen::MatrixXd& W1,
    const Eigen::VectorXd& b1,
    const Eigen::MatrixXd& W2,
    const Eigen::VectorXd& b2) {
    if (W1.rows() != 256 || W1.cols() != 784) {
        throw std::runtime_error("W1 has invalid dimensions for text save");
    }
    if (b1.size() != 256) {
        throw std::runtime_error("b1 has invalid size for text save");
    }
    if (W2.rows() != 10 || W2.cols() != 256) {
        throw std::runtime_error("W2 has invalid dimensions for text save");
    }
    if (b2.size() != 10) {
        throw std::runtime_error("b2 has invalid size for text save");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << W1.rows() << " " << W1.cols() << "\n";
    file << W2.rows() << " " << W2.cols() << "\n";
    file << b1.size() << "\n";
    file << b2.size() << "\n";

    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to write dimensions to text file: " + filename);
    }

    for (int i = 0; i < W1.rows(); ++i) {
        for (int j = 0; j < W1.cols(); ++j) {
            file << W1(i, j) << " ";
        }
        file << "\n";
    }

    for (int i = 0; i < b1.size(); ++i) {
        file << b1(i) << " ";
    }
    file << "\n";

    for (int i = 0; i < W2.rows(); ++i) {
        for (int j = 0; j < W2.cols(); ++j) {
            file << W2(i, j) << " ";
        }
        file << "\n";
    }

    for (int i = 0; i < b2.size(); ++i) {
        file << b2(i) << " ";
    }
    file << "\n";

    if (!file.good()) {
        file.close();
        throw std::runtime_error("Failed to write data to text file: " + filename);
    }

    file.close();
    return true;
}

bool WeightsIO::loadText(const std::string& filename,
    Eigen::MatrixXd& W1,
    Eigen::VectorXd& b1,
    Eigen::MatrixXd& W2,
    Eigen::VectorXd& b2) {
    if (!fileExists(filename)) {
        throw std::runtime_error("Text weights file not found: " + filename);
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    int rows1, cols1, rows2, cols2, sizeB1, sizeB2;

    if (!(file >> rows1 >> cols1)) {
        file.close();
        throw std::runtime_error("Failed to read W1 dimensions from text file: " + filename);
    }

    if (!(file >> rows2 >> cols2)) {
        file.close();
        throw std::runtime_error("Failed to read W2 dimensions from text file: " + filename);
    }

    if (!(file >> sizeB1)) {
        file.close();
        throw std::runtime_error("Failed to read b1 size from text file: " + filename);
    }

    if (!(file >> sizeB2)) {
        file.close();
        throw std::runtime_error("Failed to read b2 size from text file: " + filename);
    }

    if (rows1 != 256 || cols1 != 784) {
        file.close();
        throw std::runtime_error("W1 dimension mismatch in text file: got " +
            std::to_string(rows1) + "x" + std::to_string(cols1) +
            ", expected 256x784");
    }

    if (rows2 != 10 || cols2 != 256) {
        file.close();
        throw std::runtime_error("W2 dimension mismatch in text file: got " +
            std::to_string(rows2) + "x" + std::to_string(cols2) +
            ", expected 10x256");
    }

    if (sizeB1 != 256) {
        file.close();
        throw std::runtime_error("b1 size mismatch in text file: got " +
            std::to_string(sizeB1) + ", expected 256");
    }

    if (sizeB2 != 10) {
        file.close();
        throw std::runtime_error("b2 size mismatch in text file: got " +
            std::to_string(sizeB2) + ", expected 10");
    }

    W1.resize(rows1, cols1);
    b1.resize(sizeB1);
    W2.resize(rows2, cols2);
    b2.resize(sizeB2);

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            if (!(file >> W1(i, j))) {
                file.close();
                throw std::runtime_error("Failed to read W1 element at [" +
                    std::to_string(i) + "][" + std::to_string(j) +
                    "] from text file: " + filename);
            }
        }
    }

    for (int i = 0; i < sizeB1; ++i) {
        if (!(file >> b1(i))) {
            file.close();
            throw std::runtime_error("Failed to read b1 element at index " +
                std::to_string(i) + " from text file: " + filename);
        }
    }

    for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            if (!(file >> W2(i, j))) {
                file.close();
                throw std::runtime_error("Failed to read W2 element at [" +
                    std::to_string(i) + "][" + std::to_string(j) +
                    "] from text file: " + filename);
            }
        }
    }

    for (int i = 0; i < sizeB2; ++i) {
        if (!(file >> b2(i))) {
            file.close();
            throw std::runtime_error("Failed to read b2 element at index " +
                std::to_string(i) + " from text file: " + filename);
        }
    }

    file.close();
    return true;
}