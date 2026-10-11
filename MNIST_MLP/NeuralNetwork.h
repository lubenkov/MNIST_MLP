#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <random>

class NeuralNetwork {
private:
    Eigen::MatrixXd W1; // 256 x 784
    Eigen::VectorXd b1; // 256
    Eigen::MatrixXd W2; // 10 x 256
    Eigen::VectorXd b2; // 10

    Eigen::MatrixXd vW1, vW2;
    Eigen::VectorXd vb1, vb2;
    double momentum;

    double learningRate;
    double l2Lambda; 

    std::mt19937 rng;

    void initializeHe(Eigen::MatrixXd& weights, int fanIn, int fanOut);
    void initializeHe(Eigen::VectorXd& biases, int size);

public:
    NeuralNetwork();
    explicit NeuralNetwork(double lr);
    ~NeuralNetwork() = default;

    std::pair<Eigen::VectorXd, Eigen::VectorXd> forward(const Eigen::VectorXd& input);
    Eigen::VectorXd predict(const Eigen::VectorXd& input);
    Eigen::VectorXd softmax(const Eigen::VectorXd& z);
    double relu(double x);
    double reluDerivative(double x);

    void backward(const Eigen::VectorXd& input,
        const Eigen::VectorXd& target,
        const Eigen::VectorXd& output,
        const Eigen::VectorXd& hidden);
    void updateWeights(const Eigen::MatrixXd& dW1, const Eigen::VectorXd& db1,
        const Eigen::MatrixXd& dW2, const Eigen::VectorXd& db2);

    double trainSample(const Eigen::VectorXd& input, const Eigen::VectorXd& target);

    double crossEntropyLoss(const Eigen::VectorXd& output, const Eigen::VectorXd& target);

    void saveWeights(const std::string& filename) const;
    void loadWeights(const std::string& filename);

    void setLearningRate(double lr);
    double getLearningRate() const;
    void setMomentum(double m);

    double testAccuracy(const std::vector<std::pair<Eigen::VectorXd, int>>& testData);
};

#endif 