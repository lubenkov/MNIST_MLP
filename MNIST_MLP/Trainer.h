#ifndef TRAINER_H
#define TRAINER_H

#include <QObject>
#include <QAtomicInt>
#include "NeuralNetwork.h"
#include "DataLoader.h"
#include <vector>
#include <functional>
#include <QMutex>

class Trainer : public QObject
{
    Q_OBJECT  

private:
    NeuralNetwork* network;
    DataLoader* dataLoader;

    int numEpochs;
    double initialLearningRate;
    double finalLearningRate;
    int epochsBeforeDecay;

    mutable QMutex statsMutex;  
    std::vector<double> trainLossHistory;
    std::vector<double> trainAccuracyHistory;
    std::vector<double> testAccuracyHistory;

    QAtomicInt stopRequested;   

    void updateLearningRate(int currentEpoch);
    void printEpochStats(int epoch, double loss, double trainAcc, double testAcc);
    void printFinalStats();

public:
    explicit Trainer(NeuralNetwork* net, DataLoader* loader, QObject* parent = nullptr);
    ~Trainer() = default;

signals:
    void progressUpdate(int epoch, double loss, double trainAcc, double testAcc);
    void trainingFinished(double finalAccuracy);
    void trainingError(const QString& errorMessage);
    void progressPercent(int percent);

public slots:
    void train(int epochs = 20);

    void stop();

    double test();

    const std::vector<double>& getTrainLossHistory() const;
    const std::vector<double>& getTrainAccuracyHistory() const;
    const std::vector<double>& getTestAccuracyHistory() const;
};

#endif 