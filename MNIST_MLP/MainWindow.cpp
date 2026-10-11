#include "MainWindow.h"
#include <QGridLayout>
#include <QGroupBox>
#include <QFont>
#include <QApplication>
#include <QFileDialog>
#include <QStyle>
#include <QDebug>
#include <QIcon>
#include <QWindow>
#include <QSerialPortInfo>
#include <algorithm>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 
#endif
#include <windows.h>
#endif

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , isTrained(false)
    , weightsLoaded(false)
    , currentEpoch(0)
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

        setWindowFlag(Qt::FramelessWindowHint, true);
        setMouseTracking(true);

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
        stopTraining();
    }

    delete network;
    delete trainNetwork;
    delete dataLoader;
    

    if (arduinoPort && arduinoPort->isOpen()) {
        arduinoPort->close();
    }
}

void MainWindow::setupUI() {
    QWidget* wrapper = new QWidget(this);
    setCentralWidget(wrapper);
    QVBoxLayout* wrapperLayout = new QVBoxLayout(wrapper);
    wrapperLayout->setContentsMargins(0, 0, 0, 0);
    wrapperLayout->setSpacing(0);

    setupTitleBar(wrapperLayout);

    QWidget* centralWidget = new QWidget(wrapper);
    wrapperLayout->addWidget(centralWidget, 1);

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

    QLabel* statusValue = new QLabel("Not trained", this);
    statusValue->setStyleSheet("QLabel { color: #BF5B4D; }");
    infoLayout->addWidget(statusValue);

    QLabel* dataLabel = new QLabel("Data:", this);
    dataLabel->setFont(QFont("Arial", 10, QFont::Bold));
    infoLayout->addWidget(dataLabel);

    QLabel* dataValue = new QLabel("No data loaded", this);
    infoLayout->addWidget(dataValue);

    controlLayout->addWidget(infoGroup);

    QGroupBox* arduinoGroup = new QGroupBox("Arduino", this);
    QVBoxLayout* arduinoLayout = new QVBoxLayout(arduinoGroup);

    QHBoxLayout* arduinoPortLayout = new QHBoxLayout();
    arduinoPortCombo = new QComboBox(this);
    arduinoPortLayout->addWidget(arduinoPortCombo, 1);

    arduinoRefreshButton = new QPushButton("⟳", this);
    arduinoRefreshButton->setFixedWidth(36);
    arduinoRefreshButton->setToolTip("Обновить список портов");
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

void MainWindow::setupTitleBar(QVBoxLayout* outerLayout) {
    titleBar = new QWidget(this);
    titleBar->setFixedHeight(40);
    titleBar->setObjectName("customTitleBar");
    titleBar->setStyleSheet(
        "QWidget#customTitleBar { background-color: #F0EEE6; border-bottom: 1px solid #E0DDD0; }");
    titleBar->installEventFilter(this);

    QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins(12, 0, 0, 0);
    titleLayout->setSpacing(8);

    titleIconLabel = new QLabel(titleBar);
    titleIconLabel->setPixmap(windowIcon().pixmap(20, 20));
    titleIconLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
    titleLayout->addWidget(titleIconLabel);

    titleTextLabel = new QLabel(windowTitle(), titleBar);
    titleTextLabel->setStyleSheet("QLabel { color: #3D3929; font-weight: 600; background: transparent; }");
    titleTextLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
    titleLayout->addWidget(titleTextLabel);
    connect(this, &QWidget::windowTitleChanged, titleTextLabel, &QLabel::setText);

    titleLayout->addStretch();

    const QString capBtnStyle =
        "QPushButton { background-color: transparent; border: none; border-radius: 0px; "
        "color: #3D3929; font-weight: 600; padding: 0px; }"
        "QPushButton:hover { background-color: #E3DFD1; }"
        "QPushButton:pressed { background-color: #D9D5C7; }";

    minimizeButton = new QPushButton("—", titleBar);
    minimizeButton->setFixedSize(46, 40);
    minimizeButton->setStyleSheet(capBtnStyle);
    connect(minimizeButton, &QPushButton::clicked, this, &MainWindow::showMinimized);
    titleLayout->addWidget(minimizeButton);

    maximizeButton = new QPushButton("□", titleBar);
    maximizeButton->setFixedSize(46, 40);
    maximizeButton->setStyleSheet(capBtnStyle);
    connect(maximizeButton, &QPushButton::clicked, this, [this]() {
        if (isMaximized()) {
            showNormal();
        }
        else {
            showMaximized();
        }
        updateMaximizeIcon();
        });
    titleLayout->addWidget(maximizeButton);

    closeButton = new QPushButton("✕", titleBar);
    closeButton->setFixedSize(46, 40);
    closeButton->setStyleSheet(capBtnStyle +
        "QPushButton:hover { background-color: #BF5B4D; color: #FAF9F5; }"
        "QPushButton:pressed { background-color: #A3493D; color: #FAF9F5; }");
    connect(closeButton, &QPushButton::clicked, this, &MainWindow::close);
    titleLayout->addWidget(closeButton);

    outerLayout->addWidget(titleBar);
}

void MainWindow::updateMaximizeIcon() {
    maximizeButton->setText(isMaximized() ? "❐" : "□");
}

void MainWindow::refreshArduinoPorts() {
    QString previouslySelected = arduinoPortCombo->currentText();
    arduinoPortCombo->clear();

    const auto ports = QSerialPortInfo::availablePorts();
    for (const QSerialPortInfo& info : ports) {
        arduinoPortCombo->addItem(info.portName());
    }

    if (ports.isEmpty()) {
        arduinoPortCombo->addItem("Нет доступных портов");
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
    bool hasWeights = weightsLoaded || isTrained;

    testButton->setEnabled(hasWeights);
    recognizeButton->setEnabled(hasWeights && !drawingArea->getCanvas().isNull());

    QString status = isTrained ? "Trained" : (weightsLoaded ? "Weights Loaded" : "Not trained");
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

void MainWindow::showInfo(const QString& title, const QString& message) {
    QMessageBox::information(this, title, message);
    showStatusMessage(message, 3000);
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
    if (portName.isEmpty() || portName == "Нет доступных портов") {
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

        int digit = 0;
        double maxProb = output(0);
        for (int i = 1; i < 10; ++i) {
            if (output(i) > maxProb) {
                maxProb = output(i);
                digit = i;
            }
        }

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
   
    currentEpoch = epoch + 1;

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
        
        trainNetwork->saveWeights("temp_weights.bin");
        network->loadWeights("temp_weights.bin");
        network->saveWeights("weights.bin");

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

namespace {
    constexpr int kResizeMargin = 6; 
}

Qt::Edges MainWindow::edgeAt(const QPoint& pos) const {
    Qt::Edges edges;
    if (isMaximized()) {
        return edges; 
    }
    if (pos.x() <= kResizeMargin) edges |= Qt::LeftEdge;
    if (pos.x() >= width() - kResizeMargin) edges |= Qt::RightEdge;
    if (pos.y() <= kResizeMargin) edges |= Qt::TopEdge;
    if (pos.y() >= height() - kResizeMargin) edges |= Qt::BottomEdge;
    return edges;
}

void MainWindow::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        Qt::Edges edges = edgeAt(event->pos());
        if (edges) {
#ifdef _WIN32
            int wmszEdge = 0;
            bool left = edges & Qt::LeftEdge;
            bool right = edges & Qt::RightEdge;
            bool top = edges & Qt::TopEdge;
            bool bottom = edges & Qt::BottomEdge;

            if (left && top) wmszEdge = WMSZ_TOPLEFT;
            else if (right && top) wmszEdge = WMSZ_TOPRIGHT;
            else if (left && bottom) wmszEdge = WMSZ_BOTTOMLEFT;
            else if (right && bottom) wmszEdge = WMSZ_BOTTOMRIGHT;
            else if (left) wmszEdge = WMSZ_LEFT;
            else if (right) wmszEdge = WMSZ_RIGHT;
            else if (top) wmszEdge = WMSZ_TOP;
            else if (bottom) wmszEdge = WMSZ_BOTTOM;

            if (wmszEdge != 0) {
                ReleaseCapture();
                SendMessage(reinterpret_cast<HWND>(winId()), WM_SYSCOMMAND, SC_SIZE + wmszEdge, 0);
                event->accept();
                return;
            }
#else
            if (windowHandle()) {
                windowHandle()->startSystemResize(edges);
                event->accept();
                return;
            }
#endif
        }
    }
    QMainWindow::mousePressEvent(event);
}

void MainWindow::mouseMoveEvent(QMouseEvent* event) {
    if (!(event->buttons() & Qt::LeftButton)) {
        Qt::Edges edges = edgeAt(event->pos());
        bool diag1 = (edges & Qt::LeftEdge) && (edges & Qt::TopEdge);
        bool diag2 = (edges & Qt::RightEdge) && (edges & Qt::BottomEdge);
        bool diag3 = (edges & Qt::RightEdge) && (edges & Qt::TopEdge);
        bool diag4 = (edges & Qt::LeftEdge) && (edges & Qt::BottomEdge);

        if (diag1 || diag2) {
            setCursor(Qt::SizeFDiagCursor);
        }
        else if (diag3 || diag4) {
            setCursor(Qt::SizeBDiagCursor);
        }
        else if (edges & Qt::LeftEdge || edges & Qt::RightEdge) {
            setCursor(Qt::SizeHorCursor);
        }
        else if (edges & Qt::TopEdge || edges & Qt::BottomEdge) {
            setCursor(Qt::SizeVerCursor);
        }
        else {
            unsetCursor();
        }
    }
    QMainWindow::mouseMoveEvent(event);
}

void MainWindow::mouseReleaseEvent(QMouseEvent* event) {
    QMainWindow::mouseReleaseEvent(event);
}

void MainWindow::changeEvent(QEvent* event) {
    QMainWindow::changeEvent(event);
    if (event->type() == QEvent::WindowStateChange) {
        updateMaximizeIcon();
    }
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    if (watched == titleBar) {
        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent* me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::LeftButton) {
#ifdef _WIN32
                ReleaseCapture();
                SendMessage(reinterpret_cast<HWND>(winId()), WM_SYSCOMMAND, SC_MOVE | HTCAPTION, 0);
#else
                if (windowHandle()) {
                    windowHandle()->startSystemMove();
                }
#endif
                return true;
            }
        }
        else if (event->type() == QEvent::MouseButtonDblClick) {
            QMouseEvent* me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::LeftButton) {
                if (isMaximized()) {
                    showNormal();
                }
                else {
                    showMaximized();
                }
                updateMaximizeIcon();
                return true;
            }
        }
    }
    return QMainWindow::eventFilter(watched, event);
}