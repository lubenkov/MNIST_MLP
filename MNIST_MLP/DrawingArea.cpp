#include "DrawingArea.h"
#include "NetworkConfig.h"
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QMouseEvent>
#include <QResizeEvent>
#include <cmath>
#include <algorithm>

DrawingArea::DrawingArea(QWidget* parent)
    : QWidget(parent), penSize(14), isDrawing(false) {

    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    canvas = QImage(280, 280, QImage::Format_RGB32);
    canvas.fill(Qt::white);

    setMinimumSize(280, 280);

    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor("#F0EEE6"));
    setPalette(pal);
}

void DrawingArea::clearCanvas() {
    canvas.fill(Qt::white);
    update(); 
    emit imageChanged();
}

Eigen::VectorXd DrawingArea::getProcessedImage() const {
    if (canvas.isNull()) {
        throw std::runtime_error("Canvas is null");
    }

    Eigen::VectorXd image = preprocessImage();

    if (image.size() != NetworkConfig::kInputSize) {
        throw std::runtime_error("Processed image has invalid size: " +
            std::to_string(image.size()) + ", expected 784");
    }

    return image;
}

Eigen::VectorXd DrawingArea::imageToVector() const {
    Eigen::VectorXd vec(NetworkConfig::kInputSize);

    if (canvas.width() != 280 || canvas.height() != 280) {
        throw std::runtime_error("Canvas has invalid size: " +
            std::to_string(canvas.width()) + "x" +
            std::to_string(canvas.height()) + ", expected 280x280");
    }

    const int blockSize = 10;

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            long sum = 0;
            int count = 0;

            for (int by = 0; by < blockSize; ++by) {
                int py = y * blockSize + by;
                if (py >= canvas.height()) continue;

                for (int bx = 0; bx < blockSize; ++bx) {
                    int px = x * blockSize + bx;
                    if (px >= canvas.width()) continue;

                    QRgb pixel = canvas.pixel(px, py);
                    sum += qGray(pixel);
                    ++count;
                }
            }

            double avgGray = (count > 0) ? (static_cast<double>(sum) / count) : 255.0;
            vec(y * 28 + x) = 1.0 - (avgGray / 255.0);
        }
    }

    return vec;
}

Eigen::VectorXd DrawingArea::preprocessImage() const {
    Eigen::VectorXd raw = imageToVector();

    int minX = 28, maxX = 0, minY = 28, maxY = 0;
    bool hasPixel = false;

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            double val = raw(y * 28 + x);
            if (val > 0.1) { 
                hasPixel = true;
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }
        }
    }

    if (!hasPixel) {
        return Eigen::VectorXd::Zero(NetworkConfig::kInputSize);
    }

    int width = maxX - minX + 1;
    int height = maxY - minY + 1;

    if (width < 5 || height < 5) {
        int margin = std::max(1, (5 - std::min(width, height)) / 2);
        minX = std::max(0, minX - margin);
        maxX = std::min(27, maxX + margin);
        minY = std::max(0, minY - margin);
        maxY = std::min(27, maxY + margin);
        width = maxX - minX + 1;
        height = maxY - minY + 1;
    }

    const int targetSize = 20;
    Eigen::VectorXd cropped(targetSize * targetSize);
    cropped.setZero();

    double scale = static_cast<double>(targetSize) / std::max(width, height);
    int scaledWidth = std::max(1, std::min(targetSize, static_cast<int>(std::round(width * scale))));
    int scaledHeight = std::max(1, std::min(targetSize, static_cast<int>(std::round(height * scale))));

    int padX = (targetSize - scaledWidth) / 2;
    int padY = (targetSize - scaledHeight) / 2;

    for (int y = 0; y < scaledHeight; ++y) {
        for (int x = 0; x < scaledWidth; ++x) {
            double srcX = minX + (static_cast<double>(x) / scale);
            double srcY = minY + (static_cast<double>(y) / scale);

            int x0 = static_cast<int>(std::floor(srcX));
            int y0 = static_cast<int>(std::floor(srcY));
            int x1 = std::min(x0 + 1, 27);
            int y1 = std::min(y0 + 1, 27);

            double fx = srcX - x0;
            double fy = srcY - y0;

            double val00 = (x0 >= 0 && y0 >= 0) ? raw(y0 * 28 + x0) : 0.0;
            double val10 = (x1 >= 0 && y0 >= 0) ? raw(y0 * 28 + x1) : 0.0;
            double val01 = (x0 >= 0 && y1 >= 0) ? raw(y1 * 28 + x0) : 0.0;
            double val11 = (x1 >= 0 && y1 >= 0) ? raw(y1 * 28 + x1) : 0.0;

            double val = val00 * (1 - fx) * (1 - fy) +
                val10 * fx * (1 - fy) +
                val01 * (1 - fx) * fy +
                val11 * fx * fy;

            cropped((y + padY) * targetSize + (x + padX)) = val;
        }
    }

    Eigen::VectorXd result = Eigen::VectorXd::Zero(784);
    int offsetX = (28 - targetSize) / 2;
    int offsetY = (28 - targetSize) / 2;

    for (int y = 0; y < targetSize; ++y) {
        for (int x = 0; x < targetSize; ++x) {
            result((y + offsetY) * 28 + (x + offsetX)) = cropped(y * targetSize + x);
        }
    }

    return result;
}

void DrawingArea::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(rect(), QColor("#F0EEE6"));

    QRect targetRect = rect();
    QRect sourceRect = canvas.rect();

    double scaleX = static_cast<double>(targetRect.width()) / sourceRect.width();
    double scaleY = static_cast<double>(targetRect.height()) / sourceRect.height();
    double scale = std::min(scaleX, scaleY);

    int newWidth = static_cast<int>(sourceRect.width() * scale);
    int newHeight = static_cast<int>(sourceRect.height() * scale);
    int x = (targetRect.width() - newWidth) / 2;
    int y = (targetRect.height() - newHeight) / 2;

    QRect drawRect(x, y, newWidth, newHeight);

    QRect borderRect = drawRect.adjusted(-3, -3, 3, 3);
    painter.setPen(QPen(QColor("#E0DDD0"), 2));
    painter.setBrush(Qt::NoBrush);
    painter.drawRoundedRect(borderRect, 10, 10);

    painter.drawImage(drawRect, canvas);

    QPen gridPen(QColor(204, 120, 92, 45)); 
    gridPen.setWidth(1);
    painter.setPen(gridPen);

    for (int i = 0; i <= 28; ++i) {
        int pos = y + (i * newHeight) / 28;
        painter.drawLine(x, pos, x + newWidth, pos);
    }
    for (int i = 0; i <= 28; ++i) {
        int pos = x + (i * newWidth) / 28;
        painter.drawLine(pos, y, pos, y + newHeight);
    }
}

void DrawingArea::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isDrawing = true;
        lastPoint = event->pos();

        drawPoint(event->pos());
        update();
        emit imageChanged();
    }
}

void DrawingArea::mouseMoveEvent(QMouseEvent* event) {
    if (isDrawing && (event->buttons() & Qt::LeftButton)) {
        QPoint currentPoint = event->pos();

        drawLine(lastPoint, currentPoint);

        lastPoint = currentPoint;
        update();
        emit imageChanged();
    }
}

void DrawingArea::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && isDrawing) {
        isDrawing = false;
        emit drawingFinished();
    }
}

void DrawingArea::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    update();
}

void DrawingArea::drawPoint(const QPoint& pos) {
    QRect targetRect = rect();
    QRect sourceRect = canvas.rect();

    double scaleX = static_cast<double>(targetRect.width()) / sourceRect.width();
    double scaleY = static_cast<double>(targetRect.height()) / sourceRect.height();
    double scale = std::min(scaleX, scaleY);

    int newWidth = static_cast<int>(sourceRect.width() * scale);
    int newHeight = static_cast<int>(sourceRect.height() * scale);
    int offsetX = (targetRect.width() - newWidth) / 2;
    int offsetY = (targetRect.height() - newHeight) / 2;

    if (pos.x() < offsetX || pos.x() > offsetX + newWidth ||
        pos.y() < offsetY || pos.y() > offsetY + newHeight) {
        return;
    }

    int canvasX = static_cast<int>((pos.x() - offsetX) * sourceRect.width() / newWidth);
    int canvasY = static_cast<int>((pos.y() - offsetY) * sourceRect.height() / newHeight);

    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(QColor("#3D3929"), penSize, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawPoint(canvasX, canvasY);
}

void DrawingArea::drawLine(const QPoint& from, const QPoint& to) {
    QRect targetRect = rect();
    QRect sourceRect = canvas.rect();

    double scaleX = static_cast<double>(targetRect.width()) / sourceRect.width();
    double scaleY = static_cast<double>(targetRect.height()) / sourceRect.height();
    double scale = std::min(scaleX, scaleY);

    int newWidth = static_cast<int>(sourceRect.width() * scale);
    int newHeight = static_cast<int>(sourceRect.height() * scale);
    int offsetX = (targetRect.width() - newWidth) / 2;
    int offsetY = (targetRect.height() - newHeight) / 2;

    auto transformPoint = [&](const QPoint& p) -> QPoint {
        if (p.x() < offsetX || p.x() > offsetX + newWidth ||
            p.y() < offsetY || p.y() > offsetY + newHeight) {
            return QPoint(-1, -1); 
        }
        int canvasX = static_cast<int>((p.x() - offsetX) * sourceRect.width() / newWidth);
        int canvasY = static_cast<int>((p.y() - offsetY) * sourceRect.height() / newHeight);
        return QPoint(canvasX, canvasY);
        };

    QPoint fromCanvas = transformPoint(from);
    QPoint toCanvas = transformPoint(to);

    if (fromCanvas.x() < 0 || toCanvas.x() < 0) {
        return;
    }

    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(QColor("#3D3929"), penSize, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(fromCanvas, toCanvas);
}

QImage DrawingArea::getCanvas() const {
    return canvas;
}