#ifndef WEIGHTSIO_H
#define WEIGHTSIO_H

#include <Eigen/Dense>
#include <string>

class WeightsIO {
public:
    WeightsIO() = default;
    ~WeightsIO() = default;

    static bool saveBinary(const std::string& filename,
        const Eigen::MatrixXd& W1,
        const Eigen::VectorXd& b1,
        const Eigen::MatrixXd& W2,
        const Eigen::VectorXd& b2);

    static bool loadBinary(const std::string& filename,
        Eigen::MatrixXd& W1,
        Eigen::VectorXd& b1,
        Eigen::MatrixXd& W2,
        Eigen::VectorXd& b2);

    static bool fileExists(const std::string& filename);
};

#endif