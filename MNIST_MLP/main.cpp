#include "MainWindow.h"
#include <QtWidgets/QApplication>
#include <QFile>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    app.setStyleSheet(R"(
        QMainWindow, QWidget {
            background-color: #FAF9F5;
            color: #3D3929;
            font-family: "Segoe UI", "Arial", sans-serif;
            font-size: 10pt;
        }

        QGroupBox {
            background-color: #F0EEE6;
            border: 1px solid #E0DDD0;
            border-radius: 12px;
            margin-top: 14px;
            padding-top: 14px;
            font-weight: 600;
            color: #3D3929;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 14px;
            padding: 0 8px;
            color: #C1673F;
            background-color: transparent;
        }

        QPushButton {
            background-color: #FFFFFF;
            color: #3D3929;
            border: 1px solid #D9D5C7;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }

        QPushButton:hover {
            background-color: #FBF9F2;
            border: 1px solid #CC785C;
        }

        QPushButton:pressed {
            background-color: #F0EEE6;
        }

        QPushButton:disabled {
            background-color: #F5F4ED;
            color: #B3B0A4;
            border: 1px solid #EAE8DE;
        }

        QPushButton#trainButton {
            background-color: #CC785C;
            color: #FAF9F5;
            font-weight: 700;
            border: none;
        }

        QPushButton#trainButton:hover {
            background-color: #D98868;
        }

        QPushButton#trainButton:disabled {
            background-color: #E3C4B4;
            color: #FAF9F5;
        }

        QProgressBar {
            background-color: #F0EEE6;
            border: 1px solid #E0DDD0;
            border-radius: 8px;
            text-align: center;
            color: #3D3929;
            height: 22px;
        }

        QProgressBar::chunk {
            background-color: #CC785C;
            border-radius: 7px;
        }

        QStatusBar {
            background-color: #F0EEE6;
            color: #83827D;
            border-top: 1px solid #E0DDD0;
        }

        QLabel {
            color: #3D3929;
            background-color: transparent;
        }

        QComboBox {
            background-color: #FFFFFF;
            color: #3D3929;
            border: 1px solid #D9D5C7;
            border-radius: 8px;
            padding: 6px 10px;
        }

        QComboBox:hover {
            border: 1px solid #CC785C;
        }

        QComboBox QAbstractItemView {
            background-color: #FFFFFF;
            color: #3D3929;
            border: 1px solid #D9D5C7;
            selection-background-color: #F0EEE6;
            selection-color: #3D3929;
        }

        QMessageBox {
            background-color: #F0EEE6;
        }
    )");

    MainWindow window;
    window.show();
    return app.exec();
}