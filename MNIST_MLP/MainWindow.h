#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QProgressBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QComboBox>
#include <QSerialPort>
#include "Trainer.h"  
#include "DrawingArea.h"
#include "WeightsIO.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
    DrawingArea* drawingArea;
    QLabel* resultLabel;
    QLabel* confidenceLabel;
    QLabel* statusValue;
    QLabel* dataValue;
    QPushButton* trainButton;
    QPushButton* loadWeightsButton;
    QPushButton* testButton;
    QPushButton* clearButton;
    QPushButton* recognizeButton;
    QProgressBar* progressBar;
    QStatusBar* statusBar;

    QComboBox* arduinoPortCombo;
    QPushButton* arduinoRefreshButton;
    QPushButton* arduinoConnectButton;
    QLabel* arduinoStatusLabel;
    QSerialPort* arduinoPort;
    
    NeuralNetwork* network;      
    NeuralNetwork* trainNetwork; 
    DataLoader* dataLoader;
    Trainer* trainer;
    
    QThread* trainingThread;
    
    bool isTrained;
    bool weightsLoaded;

    void setupUI();
    void createConnections();
    void updateUIState();
    void showResult(const QString& result, double confidence);
    void stopTraining();
    void refreshArduinoPorts();
    void sendDigitToArduino(int digit);

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onTrainClicked();
    void onLoadWeightsClicked();
    void onTestClicked();
    void onClearClicked();
    void onRecognizeClicked();
    void onDrawingFinished();
    void onImageChanged();
    void onArduinoConnectClicked();

    void updateProgress(int epoch, double loss, double trainAcc, double testAcc);
    void onTrainingFinished(double finalAccuracy);
    void onTrainingError(const QString& errorMessage);
    void updateProgressPercent(int percent);

    void showError(const QString& title, const QString& message);
    void showStatusMessage(const QString& message, int timeout = 3000);

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();
};

#endif 