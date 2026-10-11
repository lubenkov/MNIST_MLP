#include "NeuralNetwork.h"
#include "WeightsIO.h"
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <iostream>

NeuralNetwork::NeuralNetwork() : NeuralNetwork(0.01) {}

NeuralNetwork::NeuralNetwork(double lr)
    : learningRate(lr), momentum(0.5), l2Lambda(0.0001), rng(std::random_device{}()) {

    initializeHe(W1, 784, 256);  
    initializeHe(b1, 256);        
    initializeHe(W2, 256, 10);    
    initializeHe(b2, 10);         

    vW1 = Eigen::MatrixXd::Zero(256, 784);
    vb1 = Eigen::VectorXd::Zero(256);
    vW2 = Eigen::MatrixXd::Zero(10, 256);
    vb2 = Eigen::VectorXd::Zero(10);
}

void NeuralNetwork::initializeHe(Eigen::MatrixXd& weights, int fanIn, int fanOut) {
    double stddev = std::sqrt(2.0 / fanIn);
    std::normal_distribution<double> dist(0.0, stddev);

    weights.resize(fanOut, fanIn);
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            weights(i, j) = dist(rng);
        }
    }
}

void NeuralNetwork::initializeHe(Eigen::VectorXd& biases, int size) {
    biases = Eigen::VectorXd::Zero(size);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> NeuralNetwork::forward(const Eigen::VectorXd& input) {
    if (input.size() != 784) {
        throw std::runtime_error("Input size must be 784, got " + std::to_string(input.size()));
    }

    Eigen::VectorXd z1 = W1 * input + b1;

    Eigen::VectorXd hidden(z1.size());
    for (int i = 0; i < z1.size(); ++i) {
        hidden(i) = relu(z1(i));
    }

    Eigen::VectorXd z2 = W2 * hidden + b2;

    return { hidden, z2 };
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd& input) {
    auto [hidden, z2] = forward(input);
    return softmax(z2);
}

Eigen::VectorXd NeuralNetwork::softmax(const Eigen::VectorXd& z) {
    double maxVal = z.maxCoeff();
    Eigen::VectorXd expZ = (z.array() - maxVal).exp();
    double sumExpZ = expZ.sum();

    if (sumExpZ == 0.0) {
        throw std::runtime_error("Softmax denominator is zero");
    }

    return expZ / sumExpZ;
}

double NeuralNetwork::relu(double x) {
    return std::max(0.0, x);
}

double NeuralNetwork::reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

void NeuralNetwork::backward(const Eigen::VectorXd& input,
    const Eigen::VectorXd& target,
    const Eigen::VectorXd& output,
    const Eigen::VectorXd& hidden) {
    if (input.size() != 784) {
        throw std::runtime_error("Input size must be 784 for backward pass");
    }
    if (target.size() != 10) {
        throw std::runtime_error("Target size must be 10 for backward pass");
    }
    if (output.size() != 10) {
        throw std::runtime_error("Output size must be 10 for backward pass");
    }
    if (hidden.size() != 256) {
        throw std::runtime_error("Hidden size must be 256 for backward pass");
    }

    Eigen::VectorXd dZ2 = output - target;

    Eigen::MatrixXd dW2 = dZ2 * hidden.transpose();
    Eigen::VectorXd db2 = dZ2;

    dW2 += l2Lambda * W2;

    Eigen::VectorXd dHidden = W2.transpose() * dZ2;

    for (int i = 0; i < dHidden.size(); ++i) {
        if (hidden(i) <= 0.0) {
            dHidden(i) = 0.0;
        }
    }

    Eigen::MatrixXd dW1 = dHidden * input.transpose();
    Eigen::VectorXd db1 = dHidden;

    dW1 += l2Lambda * W1;

    updateWeights(dW1, db1, dW2, db2);
}

void NeuralNetwork::updateWeights(const Eigen::MatrixXd& dW1, const Eigen::VectorXd& db1,
    const Eigen::MatrixXd& dW2, const Eigen::VectorXd& db2) {
    if (dW1.rows() != 256 || dW1.cols() != 784) {
        throw std::runtime_error("dW1 has invalid dimensions: " +
            std::to_string(dW1.rows()) + "x" + std::to_string(dW1.cols()));
    }
    if (db1.size() != 256) {
        throw std::runtime_error("db1 has invalid size: " + std::to_string(db1.size()));
    }
    if (dW2.rows() != 10 || dW2.cols() != 256) {
        throw std::runtime_error("dW2 has invalid dimensions: " +
            std::to_string(dW2.rows()) + "x" + std::to_string(dW2.cols()));
    }
    if (db2.size() != 10) {
        throw std::runtime_error("db2 has invalid size: " + std::to_string(db2.size()));
    }

    vW1 = momentum * vW1 - learningRate * dW1;
    vb1 = momentum * vb1 - learningRate * db1;
    vW2 = momentum * vW2 - learningRate * dW2;
    vb2 = momentum * vb2 - learningRate * db2;

    W1 += vW1;
    b1 += vb1;
    W2 += vW2;
    b2 += vb2;
}

double NeuralNetwork::trainSample(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
    if (input.size() != 784) {
        throw std::runtime_error("Input size must be 784 for training, got " +
            std::to_string(input.size()));
    }
    if (target.size() != 10) {
        throw std::runtime_error("Target size must be 10 for training, got " +
            std::to_string(target.size()));
    }

    auto [hidden, z2] = forward(input);
    Eigen::VectorXd output = softmax(z2);

    double loss = crossEntropyLoss(output, target);

    backward(input, target, output, hidden);

    return loss;
}

double NeuralNetwork::crossEntropyLoss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    if (output.size() != 10 || target.size() != 10) {
        throw std::runtime_error("Both output and target must have size 10 for cross-entropy");
    }

    const double epsilon = 1e-15;
    double loss = 0.0;

    for (int i = 0; i < 10; ++i) {
        if (target(i) > 0.5) {  
            double pred = std::max(epsilon, std::min(1.0 - epsilon, output(i)));
            loss -= std::log(pred);
        }
    }

    return loss;
}

void NeuralNetwork::saveWeights(const std::string& filename) const {

    if (!WeightsIO::saveBinary(filename, W1, b1, W2, b2)) {
        throw std::runtime_error("Failed to save weights to: " + filename);
    }
}

void NeuralNetwork::loadWeights(const std::string& filename) {

    if (!WeightsIO::loadBinary(filename, W1, b1, W2, b2)) {
        throw std::runtime_error("Failed to load weights from: " + filename);
    }
}

void NeuralNetwork::setLearningRate(double lr) {
    if (lr <= 0.0) {
        throw std::runtime_error("Learning rate must be positive, got " + std::to_string(lr));
    }
    learningRate = lr;
}

double NeuralNetwork::getLearningRate() const {
    return learningRate;
}

void NeuralNetwork::setMomentum(double m) {
    if (m < 0.0 || m >= 1.0) {
        throw std::runtime_error("Momentum must be in range [0, 1), got " + std::to_string(m));
    }
    momentum = m;
}

double NeuralNetwork::testAccuracy(const std::vector<std::pair<Eigen::VectorXd, int>>& testData) {
    if (testData.empty()) {
        throw std::runtime_error("Test data is empty");
    }

    int correct = 0;
    int total = 0;

    for (const auto& sample : testData) {
        const Eigen::VectorXd& image = sample.first;
        int expectedLabel = sample.second;

        if (image.size() != 784) {
            throw std::runtime_error("Test image has invalid size: " + std::to_string(image.size()));
        }

        Eigen::VectorXd output = predict(image);

        int predictedLabel = 0;
        double maxProb = output(0);
        for (int i = 1; i < 10; ++i) {
            if (output(i) > maxProb) {
                maxProb = output(i);
                predictedLabel = i;
            }
        }

        if (predictedLabel == expectedLabel) {
            correct++;
        }
        total++;
    }

    return static_cast<double>(correct) / total;
}