#include "WeightsIO.h"
#include "NetworkConfig.h"
#include <fstream>
#include <cstdint>


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
    if (W1.rows() != NetworkConfig::kHiddenSize || W1.cols() != NetworkConfig::kInputSize) {
        throw std::runtime_error("W1 has invalid dimensions: " +
            std::to_string(W1.rows()) + "x" + std::to_string(W1.cols()) +
            ", expected 256x784");
    }

    if (b1.size() != NetworkConfig::kHiddenSize) {
        throw std::runtime_error("b1 has invalid size: " +
            std::to_string(b1.size()) + ", expected 256");
    }

    if (W2.rows() != NetworkConfig::kOutputSize || W2.cols() != NetworkConfig::kHiddenSize) {
        throw std::runtime_error("W2 has invalid dimensions: " +
            std::to_string(W2.rows()) + "x" + std::to_string(W2.cols()) +
            ", expected 10x256");
    }

    if (b2.size() != NetworkConfig::kOutputSize) {
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

    if (rows1 != NetworkConfig::kHiddenSize || cols1 != NetworkConfig::kInputSize) {
        file.close();
        throw std::runtime_error("W1 dimension mismatch in file: got " +
            std::to_string(rows1) + "x" + std::to_string(cols1) +
            ", expected 256x784");
    }

    if (sizeB1 != NetworkConfig::kHiddenSize) {
        file.close();
        throw std::runtime_error("b1 size mismatch in file: got " +
            std::to_string(sizeB1) + ", expected 256");
    }

    if (rows2 != NetworkConfig::kOutputSize || cols2 != NetworkConfig::kHiddenSize) {
        file.close();
        throw std::runtime_error("W2 dimension mismatch in file: got " +
            std::to_string(rows2) + "x" + std::to_string(cols2) +
            ", expected 10x256");
    }

    if (sizeB2 != NetworkConfig::kOutputSize) {
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

