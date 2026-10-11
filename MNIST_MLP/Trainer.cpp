#include "Trainer.h"
#include <QThread>
#include <QDebug>

Trainer::Trainer(NeuralNetwork* net, DataLoader* loader, QObject* parent)
    : QObject(parent)
    , network(net)
    , dataLoader(loader)
    , numEpochs(20)
    , initialLearningRate(0.01)
    , finalLearningRate(0.001)
    , epochsBeforeDecay(10)
    , stopRequested(0)  
{
    if (!network || !dataLoader) {
        throw std::runtime_error("Network or DataLoader is null");
    }

}

void Trainer::stop()
{
    qDebug() << "Trainer: Stop requested";
    stopRequested.store(1);  
}

void Trainer::train(int epochs)
{
    stopRequested.store(0);

    qDebug() << "Trainer: Training started in thread:" << QThread::currentThreadId();

    try {
        if (stopRequested.load()) {
            emit trainingError("Training cancelled before start");
            return;
        }

        numEpochs = epochs;

        if (dataLoader->getTrainSize() == 0) {
            dataLoader->loadTrainData("resources/lib_MNIST_edit.txt");
            dataLoader->loadTestData("resources/lib_10k.txt");
        }

        if (dataLoader->getTrainSize() == 0) {
            emit trainingError("No training data loaded");
            return;
        }

        dataLoader->shuffleTrainData();
        const auto& trainData = dataLoader->getTrainData();

        double lastTestAccuracy = 0.0;

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            if (stopRequested.load()) {
                qDebug() << "Trainer: Stop requested at epoch" << epoch;
                emit trainingFinished(0.0);  
                return;
            }

            updateLearningRate(epoch);

            double totalLoss = 0.0;
            int correctPredictions = 0;
            int totalSamples = 0;
            const int datasetSize = static_cast<int>(trainData.size());

            for (const auto& sample : trainData) {
                if (totalSamples % 1000 == 0 && stopRequested.load()) {
                    qDebug() << "Trainer: Stop requested mid-epoch";
                    emit trainingFinished(0.0);
                    return;
                }

                Eigen::VectorXd target = dataLoader->labelToOneHot(sample.label);
                double loss = network->trainSample(sample.image, target);
                totalLoss += loss;

                Eigen::VectorXd prediction = network->predict(sample.image);
                int predictedLabel = dataLoader->oneHotToLabel(prediction);
                if (predictedLabel == sample.label) {
                    correctPredictions++;
                }
                totalSamples++;

                if (totalSamples % 200 == 0) {
                    double epochFraction = static_cast<double>(totalSamples) / datasetSize;
                    double overallFraction = (epoch + epochFraction) / numEpochs;
                    emit progressPercent(static_cast<int>(overallFraction * 100));
                }
            }

            double avgLoss = totalLoss / totalSamples;
            double trainAccuracy = static_cast<double>(correctPredictions) / totalSamples;

            double testAccuracy = 0.0;
            try {
                testAccuracy = test();
            }
            catch (...) {
                testAccuracy = 0.0;
            }

            lastTestAccuracy = testAccuracy;

            emit progressUpdate(epoch, avgLoss, trainAccuracy, testAccuracy);

            qDebug() << QString("Epoch %1/%2 | Loss: %3 | Train: %4% | Test: %5%")
                .arg(epoch + 1)
                .arg(numEpochs)
                .arg(avgLoss, 0, 'f', 4)
                .arg(static_cast<int>(trainAccuracy * 100))
                .arg(static_cast<int>(testAccuracy * 100));
        }

        emit trainingFinished(lastTestAccuracy);

        qDebug() << "Trainer: Training completed successfully";

    }
    catch (const std::exception& e) {
        QString errorMsg = QString::fromStdString(e.what());
        qDebug() << "Trainer: Error:" << errorMsg;
        emit trainingError(errorMsg);
    }
}

double Trainer::test()
{
    if (dataLoader->getTestSize() == 0) {
        throw std::runtime_error("No test data loaded");
    }

    const auto& testData = dataLoader->getTestData();

    std::vector<std::pair<Eigen::VectorXd, int>> testSamples;
    testSamples.reserve(testData.size());

    for (const auto& sample : testData) {
        testSamples.push_back({ sample.image, sample.label });
    }

    return network->testAccuracy(testSamples);
}

void Trainer::updateLearningRate(int currentEpoch)
{
    if (currentEpoch >= epochsBeforeDecay) {
        double progress = static_cast<double>(currentEpoch - epochsBeforeDecay) /
            (numEpochs - epochsBeforeDecay);
        double newLR = initialLearningRate - progress * (initialLearningRate - finalLearningRate);
        newLR = std::max(finalLearningRate, newLR);
        network->setLearningRate(newLR);
    }
    else {
        network->setLearningRate(initialLearningRate);
    }
}
