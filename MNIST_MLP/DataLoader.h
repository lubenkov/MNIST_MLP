#ifndef DATALOADER_H
#define DATALOADER_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <utility>


struct MNISTSample{
    Eigen::VectorXd image;  
    int label;              
};


class DataLoader {
private:
    std::vector<MNISTSample> trainData;
    std::vector<MNISTSample> testData;

    Eigen::VectorXd normalizeImage(const std::vector<double>& pixels);
    std::vector<double> parseLine(const std::string& line);

public:
    DataLoader() = default;
    ~DataLoader() = default;

    bool loadTrainData(const std::string& filename);
    bool loadTestData(const std::string& filename);

    Eigen::VectorXd labelToOneHot(int label) const;
    int oneHotToLabel(const Eigen::VectorXd& oneHot) const;

    const std::vector<MNISTSample>& getTrainData() const;
    const std::vector<MNISTSample>& getTestData() const;
    size_t getTrainSize() const;
    size_t getTestSize() const;

    void shuffleTrainData();
};

#endif 