#ifndef TRAINER_H
#define TRAINER_H

#include <QObject>
#include <QAtomicInt>
#include "NeuralNetwork.h"
#include "DataLoader.h"

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

    QAtomicInt stopRequested;

    void updateLearningRate(int currentEpoch);

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
};

#endif