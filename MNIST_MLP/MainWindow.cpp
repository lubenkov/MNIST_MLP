#include "MainWindow.h"
#include <QGroupBox>
#include <QFont>
#include <QApplication>
#include <QStyle>
#include <QDebug>
#include <QIcon>
#include <QSerialPortInfo>
#include <algorithm>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , isTrained(false)
    , weightsLoaded(false)
    , trainingThread(nullptr)
{
    try {
        network = new NeuralNetwork(0.001);        
        trainNetwork = new NeuralNetwork(0.001);   
        dataLoader = new DataLoader();

        trainer = new Trainer(trainNetwork, dataLoader);

        connect(trainer, &Trainer::progressUpdate,
            this, &MainWindow::updateProgress);
        connect(trainer, &Trainer::trainingFinished,
            this, &MainWindow::onTrainingFinished);
        connect(trainer, &Trainer::trainingError,
            this, &MainWindow::onTrainingError);
        connect(trainer, &Trainer::progressPercent,
            this, &MainWindow::updateProgressPercent);

        trainingThread = new QThread(this);

        trainer->moveToThread(trainingThread);

        connect(trainingThread, &QThread::started,
            trainer, [this]() {
                trainer->train(25);  
            }, Qt::QueuedConnection);

        
        setWindowIcon(QIcon("resources/app_icon.svg"));

        setupUI();
        createConnections();
        updateUIState();
        showStatusMessage("Ready. Load weights or train network.");

        setMinimumSize(900, 650);
        resize(1000, 720);
    }
    catch (const std::exception& e) {
        showError("Initialization Error", QString::fromStdString(e.what()));
    }
}

MainWindow::~MainWindow()
{
    if (trainingThread && trainingThread->isRunning()) {
        trainer->stop();
        trainingThread->quit();
        trainingThread->wait();
    }

    delete trainer;
    delete network;
    delete trainNetwork;
    delete dataLoader;

    if (arduinoPort && arduinoPort->isOpen()) {
        arduinoPort->close();
    }
}

void MainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    QGroupBox* drawingGroup = new QGroupBox("Draw a Digit", this);
    QVBoxLayout* drawingLayout = new QVBoxLayout(drawingGroup);

    drawingArea = new DrawingArea(this);
    drawingArea->setMinimumSize(280, 280);
    drawingArea->setMaximumSize(400, 400);
    drawingArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    drawingLayout->addWidget(drawingArea);

    QHBoxLayout* resultLayout = new QHBoxLayout();
    resultLayout->setContentsMargins(6, 4, 6, 4);
    resultLayout->setSpacing(12);
    QLabel* resultLabelText = new QLabel("Prediction:", this);
    resultLabelText->setFont(QFont("Arial", 12, QFont::Bold));
    resultLayout->addWidget(resultLabelText);

    resultLabel = new QLabel("?", this);
    resultLabel->setFont(QFont("Arial", 70, QFont::Bold));
    resultLabel->setAlignment(Qt::AlignCenter);
    resultLabel->setStyleSheet("QLabel { color: #3D3929; background-color: #F0EEE6; border: 3px solid #CC785C; border-radius: 12px; padding: 5px; font-size: 70pt; font-weight: 700; }");
    resultLabel->setMinimumHeight(150);
    resultLayout->addWidget(resultLabel);

    drawingLayout->addLayout(resultLayout);

    QHBoxLayout* confidenceLayout = new QHBoxLayout();
    confidenceLayout->setContentsMargins(6, 0, 6, 4);
    confidenceLayout->setSpacing(8);
    QLabel* confidenceLabelText = new QLabel("Confidence:", this);
    confidenceLabelText->setFont(QFont("Arial", 10));
    confidenceLayout->addWidget(confidenceLabelText);

    confidenceLabel = new QLabel("0%", this);
    confidenceLabel->setFont(QFont("Arial", 10));
    confidenceLabel->setStyleSheet("QLabel { color: #5A8F5A; }");
    confidenceLayout->addWidget(confidenceLabel);
    confidenceLayout->addStretch();

    drawingLayout->addLayout(confidenceLayout);

    QHBoxLayout* drawButtonsLayout = new QHBoxLayout();
    clearButton = new QPushButton("Clear", this);
    clearButton->setIcon(style()->standardIcon(QStyle::SP_DialogResetButton));
    recognizeButton = new QPushButton("Recognize", this);
    recognizeButton->setIcon(style()->standardIcon(QStyle::SP_DialogApplyButton));
    recognizeButton->setEnabled(false);

    drawButtonsLayout->addWidget(clearButton);
    drawButtonsLayout->addWidget(recognizeButton);
    drawingLayout->addLayout(drawButtonsLayout);

    mainLayout->addWidget(drawingGroup, 1);

    QGroupBox* controlGroup = new QGroupBox("Controls", this);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlGroup);

    trainButton = new QPushButton("Train Network", this);
    trainButton->setObjectName("trainButton");
    trainButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    trainButton->setMinimumHeight(44);
    controlLayout->addWidget(trainButton);

    loadWeightsButton = new QPushButton("Load Weights", this);
    loadWeightsButton->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
    loadWeightsButton->setMinimumHeight(40);
    controlLayout->addWidget(loadWeightsButton);

    testButton = new QPushButton("Test on MNIST", this);
    testButton->setIcon(style()->standardIcon(QStyle::SP_FileDialogContentsView));
    testButton->setMinimumHeight(40);
    testButton->setEnabled(false);
    controlLayout->addWidget(testButton);

    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setTextVisible(true);
    progressBar->setFormat("%p%");
    controlLayout->addWidget(progressBar);

    QGroupBox* infoGroup = new QGroupBox("Status", this);
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);

    QLabel* statusLabel = new QLabel("Network Status:", this);
    statusLabel->setFont(QFont("Arial", 10, QFont::Bold));
    infoLayout->addWidget(statusLabel);

    statusValue = new QLabel("Not trained", this);
    statusValue->setStyleSheet("QLabel { color: #BF5B4D; }");
    infoLayout->addWidget(statusValue);

    QLabel* dataLabel = new QLabel("Data:", this);
    dataLabel->setFont(QFont("Arial", 10, QFont::Bold));
    infoLayout->addWidget(dataLabel);

    dataValue = new QLabel("No data loaded", this);
    infoLayout->addWidget(dataValue);

    controlLayout->addWidget(infoGroup);

    QGroupBox* arduinoGroup = new QGroupBox("Arduino", this);
    QVBoxLayout* arduinoLayout = new QVBoxLayout(arduinoGroup);

    QHBoxLayout* arduinoPortLayout = new QHBoxLayout();
    arduinoPortCombo = new QComboBox(this);
    arduinoPortLayout->addWidget(arduinoPortCombo, 1);

    arduinoRefreshButton = new QPushButton("⟳", this);
    arduinoRefreshButton->setFixedWidth(36);
    arduinoRefreshButton->setToolTip("Refresh port list");
    arduinoPortLayout->addWidget(arduinoRefreshButton);

    arduinoLayout->addLayout(arduinoPortLayout);

    arduinoConnectButton = new QPushButton("Connect", this);
    arduinoLayout->addWidget(arduinoConnectButton);

    arduinoStatusLabel = new QLabel("Not connected", this);
    arduinoStatusLabel->setStyleSheet("QLabel { color: #BF5B4D; }");
    arduinoLayout->addWidget(arduinoStatusLabel);

    controlLayout->addWidget(arduinoGroup);

    controlLayout->addStretch();
    mainLayout->addWidget(controlGroup, 0);

    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);

    arduinoPort = new QSerialPort(this);
    refreshArduinoPorts();
}

void MainWindow::refreshArduinoPorts() {
    QString previouslySelected = arduinoPortCombo->currentText();
    arduinoPortCombo->clear();

    const auto ports = QSerialPortInfo::availablePorts();
    for (const QSerialPortInfo& info : ports) {
        arduinoPortCombo->addItem(info.portName());
    }

    if (ports.isEmpty()) {
        arduinoPortCombo->addItem("No ports available");
        arduinoConnectButton->setEnabled(false);
    }
    else {
        arduinoConnectButton->setEnabled(true);
        int idx = arduinoPortCombo->findText(previouslySelected);
        if (idx >= 0) {
            arduinoPortCombo->setCurrentIndex(idx);
        }
    }
}

void MainWindow::sendDigitToArduino(int digit) {
    if (!arduinoPort || !arduinoPort->isOpen()) {
        return; 
    }

    digit = std::max(0, std::min(9, digit));
    QByteArray data = QByteArray::number(digit) + "\n";
    arduinoPort->write(data);
}

void MainWindow::createConnections() {
    connect(trainButton, &QPushButton::clicked, this, &MainWindow::onTrainClicked);
    connect(loadWeightsButton, &QPushButton::clicked, this, &MainWindow::onLoadWeightsClicked);
    connect(testButton, &QPushButton::clicked, this, &MainWindow::onTestClicked);
    connect(clearButton, &QPushButton::clicked, this, &MainWindow::onClearClicked);
    connect(recognizeButton, &QPushButton::clicked, this, &MainWindow::onRecognizeClicked);

    connect(drawingArea, &DrawingArea::drawingFinished, this, &MainWindow::onDrawingFinished);
    connect(drawingArea, &DrawingArea::imageChanged, this, &MainWindow::onImageChanged);

    connect(arduinoRefreshButton, &QPushButton::clicked, this, &MainWindow::refreshArduinoPorts);
    connect(arduinoConnectButton, &QPushButton::clicked, this, &MainWindow::onArduinoConnectClicked);
}

void MainWindow::updateUIState() {
    const bool hasWeights = weightsLoaded || isTrained;

    testButton->setEnabled(hasWeights);
    recognizeButton->setEnabled(hasWeights && !drawingArea->getCanvas().isNull());

    QString status;
    QString statusColor;
    if (isTrained) {
        status = "Trained";
        statusColor = "#5A8F5A";
    }
    else if (weightsLoaded) {
        status = "Weights Loaded";
        statusColor = "#C48A3B";
    }
    else {
        status = "Not trained";
        statusColor = "#BF5B4D";
    }

    statusValue->setText(status);
    statusValue->setStyleSheet(QString("QLabel { color: %1; }").arg(statusColor));

    const size_t trainSize = dataLoader->getTrainSize();
    const size_t testSize = dataLoader->getTestSize();
    if (trainSize == 0) {
        dataValue->setText("No data loaded");
    }
    else if (testSize == 0) {
        dataValue->setText(QString("%1 train samples").arg(trainSize));
    }
    else {
        dataValue->setText(QString("%1 train / %2 test samples").arg(trainSize).arg(testSize));
    }

    setWindowTitle(QString("Digit Recognizer - [%1]").arg(status));
}

void MainWindow::showResult(const QString& result, double confidence) {
    resultLabel->setText(result);

    QString color;
    if (confidence >= 0.8) {
        color = "#5A8F5A"; 
    }
    else if (confidence >= 0.5) {
        color = "#C48A3B"; 
    }
    else {
        color = "#BF5B4D";
    }

    resultLabel->setStyleSheet(QString("QLabel { color: %1; background-color: #F0EEE6; "
        "border: 3px solid %1; border-radius: 12px; "
        "padding: 5px; font-size: 70pt; font-weight: 700; }").arg(color));

    confidenceLabel->setText(QString("%1%").arg(static_cast<int>(confidence * 100)));
}

void MainWindow::showStatusMessage(const QString& message, int timeout) {
    if (statusBar) {
        statusBar->showMessage(message, timeout);
    }
}

void MainWindow::showError(const QString& title, const QString& message) {
    QMessageBox::critical(this, title, message);
    showStatusMessage("Error: " + message, 5000);
}

void MainWindow::onTrainClicked()
{
    if (trainingThread && trainingThread->isRunning()) {
        stopTraining();
        return;
    }

    loadWeightsButton->setEnabled(false);
    testButton->setEnabled(false);
    recognizeButton->setEnabled(false);
    progressBar->setValue(0);

    trainButton->setText("Stop Training");
    trainButton->setStyleSheet(
        "QPushButton#trainButton { background-color: #BF5B4D; color: #FAF9F5; font-weight: 700; border: none; }");

    showStatusMessage("Starting training...");

    trainingThread->start();
}

void MainWindow::onLoadWeightsClicked() {
    try {
        if (!WeightsIO::fileExists("weights.bin")) {
            QMessageBox::information(this, "Weights Not Found",
                "No weights file found.\n\n"
                "Please train the network first or place weights.bin in the application directory.");
            return;
        }

        showStatusMessage("Loading weights from weights.bin...");
        QApplication::processEvents();

        network->loadWeights("weights.bin");
        trainNetwork->loadWeights("weights.bin");

        weightsLoaded = true;
        isTrained = false;

        showStatusMessage("Weights loaded successfully!");
        QMessageBox::information(this, "Weights Loaded",
            "Weights loaded successfully from weights.bin");

    }
    catch (const std::exception& e) {
        showError("Error", QString::fromStdString(e.what()));
        weightsLoaded = false;
    }

    updateUIState();
}

void MainWindow::onTestClicked() {
    if (!weightsLoaded && !isTrained) {
        showError("Not Ready", "Please train the network or load weights first.");
        return;
    }

    try {
        showStatusMessage("Loading test data...");
        QApplication::processEvents();

        if (dataLoader->getTestSize() == 0) {
            dataLoader->loadTestData("resources/lib_10k.txt");
        }

        showStatusMessage("Testing on 10,000 MNIST samples...");
        QApplication::processEvents();

        double accuracy = trainer->test();

        showStatusMessage(QString("Test completed! Accuracy: %1%")
            .arg(static_cast<int>(accuracy * 100)));

        QMessageBox::information(this, "Test Results",
            QString("MNIST Test Results\n\n"
                "Test samples: 10,000\n"
                "Accuracy: %1%\n"
                "Correct: %2\n"
                "Incorrect: %3")
            .arg(static_cast<int>(accuracy * 100))
            .arg(static_cast<int>(accuracy * 10000))
            .arg(static_cast<int>((1 - accuracy) * 10000)));
        updateUIState();

    }
    catch (const std::exception& e) {
        showError("Error", QString::fromStdString(e.what()));
    }
}

void MainWindow::onArduinoConnectClicked() {
    if (arduinoPort->isOpen()) {
        arduinoPort->close();
        arduinoConnectButton->setText("Connect");
        arduinoStatusLabel->setText("Not connected");
        arduinoStatusLabel->setStyleSheet("QLabel { color: #BF5B4D; }");
        showStatusMessage("Arduino disconnected");
        return;
    }

    QString portName = arduinoPortCombo->currentText();
    if (portName.isEmpty() || portName == "No ports available") {
        showError("Arduino", "No serial port selected. Connect the Arduino and press refresh.");
        return;
    }

    arduinoPort->setPortName(portName);
    arduinoPort->setBaudRate(QSerialPort::Baud9600);
    arduinoPort->setDataBits(QSerialPort::Data8);
    arduinoPort->setParity(QSerialPort::NoParity);
    arduinoPort->setStopBits(QSerialPort::OneStop);
    arduinoPort->setFlowControl(QSerialPort::NoFlowControl);

    if (arduinoPort->open(QIODevice::ReadWrite)) {
        arduinoConnectButton->setText("Disconnect");
        arduinoStatusLabel->setText(QString("Connected: %1").arg(portName));
        arduinoStatusLabel->setStyleSheet("QLabel { color: #5A8F5A; }");
        showStatusMessage(QString("Arduino connected on %1").arg(portName));
    }
    else {
        showError("Arduino", QString("Failed to open port %1: %2")
            .arg(portName)
            .arg(arduinoPort->errorString()));
    }
}

void MainWindow::onClearClicked() {
    drawingArea->clearCanvas();
    resultLabel->setText("?");
    resultLabel->setStyleSheet("QLabel { color: #3D3929; background-color: #F0EEE6; "
        "border: 3px solid #CC785C; border-radius: 12px; "
        "padding: 5px; font-size: 70pt; font-weight: 700; }");
    confidenceLabel->setText("0%");
    sendDigitToArduino(0); 
    showStatusMessage("Canvas cleared");
}

void MainWindow::onRecognizeClicked() {
    if (!weightsLoaded && !isTrained) {
        showError("Not Ready", "Please train the network or load weights first.");
        return;
    }

    try {
        showStatusMessage("Recognizing digit...");
        QApplication::processEvents();

        Eigen::VectorXd image = drawingArea->getProcessedImage();
        Eigen::VectorXd output = network->predict(image);

        const int digit = NeuralNetwork::argmax(output);
        const double maxProb = output(digit);

        showResult(QString::number(digit), maxProb);
        sendDigitToArduino(digit);
        showStatusMessage(QString("Recognized digit: %1 (confidence: %2%)")
            .arg(digit)
            .arg(static_cast<int>(maxProb * 100)));

    }
    catch (const std::exception& e) {
        showError("Error", QString::fromStdString(e.what()));
    }
}



void MainWindow::onDrawingFinished() {
    if (weightsLoaded || isTrained) {
        onRecognizeClicked();
    }
}

void MainWindow::onImageChanged() {
    recognizeButton->setEnabled((weightsLoaded || isTrained) &&
        !drawingArea->getCanvas().isNull());
}


void MainWindow::updateProgress(int epoch, double loss, double trainAcc, double testAcc)
{
    QString status = QString("Epoch %1/5 | Loss: %2 | Train: %3% | Test: %4%")
        .arg(epoch + 1)
        .arg(loss, 0, 'f', 4)
        .arg(static_cast<int>(trainAcc * 100))
        .arg(static_cast<int>(testAcc * 100));

    showStatusMessage(status);
}

void MainWindow::onTrainingFinished(double finalAccuracy)
{
    qDebug() << "Training finished";

    
    trainButton->setText("Train Network");
    trainButton->setStyleSheet("");  
    trainButton->setEnabled(true);
    loadWeightsButton->setEnabled(true);

    if (finalAccuracy > 0.001) {
        
        trainNetwork->saveWeights("weights.bin");
        network->loadWeights("weights.bin");

        isTrained = true;
        weightsLoaded = true;

        progressBar->setValue(100);
        updateUIState();

        showStatusMessage(QString("Training completed! Accuracy: %1%")
            .arg(static_cast<int>(finalAccuracy * 100)));

        QMessageBox::information(this, "Training Complete",
            QString("Training completed!\nFinal Accuracy: %1%")
            .arg(static_cast<int>(finalAccuracy * 100)));
        
        trainingThread->quit();
        
    }
    else {
        
        progressBar->setValue(0);
        showStatusMessage("Training stopped by user");
        
        trainingThread->quit();
        
    }
}

void MainWindow::updateProgressPercent(int percent)
{
    progressBar->setValue(percent);
}

void MainWindow::onTrainingError(const QString& errorMessage)
{
    qDebug() << "Training error:" << errorMessage;

    trainButton->setText("Train Network");
    trainButton->setStyleSheet("");  
    trainButton->setEnabled(true);
    loadWeightsButton->setEnabled(true);
    updateUIState();
    progressBar->setValue(0);

    showError("Training Error", errorMessage);
}


void MainWindow::stopTraining()
{
    if (trainer && trainingThread && trainingThread->isRunning()) {
        qDebug() << "Stopping training...";

        trainer->stop();

        trainButton->setText("Train Network");
        trainButton->setStyleSheet("");  
        trainButton->setEnabled(true);
        loadWeightsButton->setEnabled(true);
        updateUIState();
        progressBar->setValue(0);

        showStatusMessage("Training stopped");
    }
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    qDebug() << "Closing window, stopping training if running...";

    if (trainingThread && trainingThread->isRunning()) {
        trainer->stop();          
        trainingThread->quit();
        trainingThread->wait();
    }

    event->accept();
    QMainWindow::closeEvent(event);
}