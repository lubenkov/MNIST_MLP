#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cctype>

bool DataLoader::loadTrainData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open training data file: " + filename);
    }

    trainData.clear();
    std::string line;
    bool isDataSection = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        if (!isDataSection) {
            bool hasLetters = false;
            for (char c : line) {
                if (std::isalpha(static_cast<unsigned char>(c))) {
                    hasLetters = true;
                    break;
                }
            }

            if (hasLetters) {
                continue;
            }

            bool hasWhitespace = line.find_first_of(" \t") != std::string::npos;
            if (!hasWhitespace) {
                continue;
            }

            isDataSection = true;
        }

        std::vector<double> values = parseLine(line);

        if (values.empty()) {
            continue;
        }

        if (values.size() != 785) {
            continue;
        }

        MNISTSample sample;
        sample.label = static_cast<int>(values[0]);

        if (sample.label < 0 || sample.label > 9) {
            continue;
        }

        std::vector<double> pixels(values.begin() + 1, values.end());
        sample.image = normalizeImage(pixels);

        trainData.push_back(sample);
    }

    file.close();

    if (trainData.empty()) {
        throw std::runtime_error("No training data loaded from file: " + filename);
    }

    return true;
}

bool DataLoader::loadTestData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open test data file: " + filename);
    }

    testData.clear();
    std::string line;
    bool isDataSection = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        if (!isDataSection) {
            bool hasLetters = false;
            for (char c : line) {
                if (std::isalpha(static_cast<unsigned char>(c))) {
                    hasLetters = true;
                    break;
                }
            }

            if (hasLetters) {
                continue;
            }

            bool hasWhitespace = line.find_first_of(" \t") != std::string::npos;
            if (!hasWhitespace) {
                continue;
            }

            isDataSection = true;
        }

        std::vector<double> values = parseLine(line);

        if (values.empty()) {
            continue;
        }

        if (values.size() != 785) {
            continue;
        }

        MNISTSample sample;
        sample.label = static_cast<int>(values[0]);

        if (sample.label < 0 || sample.label > 9) {
            continue;
        }

        std::vector<double> pixels(values.begin() + 1, values.end());
        sample.image = normalizeImage(pixels);

        testData.push_back(sample);
    }

    file.close();

    if (testData.empty()) {
        throw std::runtime_error("No test data loaded from file: " + filename);
    }

    return true;
}

Eigen::VectorXd DataLoader::normalizeImage(const std::vector<double>& pixels) {
    if (pixels.size() != 784) {
        throw std::runtime_error("Expected 784 pixels, got " + std::to_string(pixels.size()));
    }

    Eigen::VectorXd normalized(784);

    for (int i = 0; i < 784; ++i) {
        double pixel = pixels[i];

        if (pixel >= 0.0 && pixel <= 1.0) {
            normalized(i) = pixel;
        }
        else if (pixel > 1.0 && pixel <= 255.0) {
            normalized(i) = pixel / 255.0;
        }
        else {
            normalized(i) = std::max(0.0, std::min(1.0, pixel / 255.0));
        }
    }

    return normalized;
}

std::vector<double> DataLoader::parseLine(const std::string& line) {
    std::vector<double> values;
    std::istringstream iss(line);
    std::string token;

    while (iss >> token) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);

        if (token.empty()) continue;

        bool isNumber = true;
        bool hasDigit = false;
        for (char c : token) {
            if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == 'e' || c == 'E') {
                if (std::isdigit(static_cast<unsigned char>(c))) hasDigit = true;
            }
            else {
                isNumber = false;
                break;
            }
        }

        if (!isNumber || !hasDigit) {
            continue;
        }

        try {
            double value = std::stod(token);
            values.push_back(value);
        }
        catch (const std::exception& e) {
            continue;
        }
    }

    return values;
}

Eigen::VectorXd DataLoader::labelToOneHot(int label) const {
    if (label < 0 || label > 9) {
        throw std::runtime_error("Invalid label for one-hot encoding: " + std::to_string(label));
    }

    Eigen::VectorXd oneHot = Eigen::VectorXd::Zero(10);
    oneHot(label) = 1.0;
    return oneHot;
}

int DataLoader::oneHotToLabel(const Eigen::VectorXd& oneHot) const {
    if (oneHot.size() != 10) {
        throw std::runtime_error("oneHot vector size must be 10, got " + std::to_string(oneHot.size()));
    }

    int maxIndex = 0;
    double maxValue = oneHot(0);
    for (int i = 1; i < 10; ++i) {
        if (oneHot(i) > maxValue) {
            maxValue = oneHot(i);
            maxIndex = i;
        }
    }

    return maxIndex;
}

const std::vector<MNISTSample>& DataLoader::getTrainData() const {
    if (trainData.empty()) {
        throw std::runtime_error("Training data is empty. Load data first.");
    }
    return trainData;
}

const std::vector<MNISTSample>& DataLoader::getTestData() const {
    if (testData.empty()) {
        throw std::runtime_error("Test data is empty. Load data first.");
    }
    return testData;
}

size_t DataLoader::getTrainSize() const {
    return trainData.size();
}

size_t DataLoader::getTestSize() const {
    return testData.size();
}

void DataLoader::shuffleTrainData() {
    if (trainData.empty()) {
        throw std::runtime_error("Cannot shuffle empty training data");
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(trainData.begin(), trainData.end(), std::default_random_engine(seed));
}
