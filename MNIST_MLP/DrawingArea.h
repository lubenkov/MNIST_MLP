#ifndef DRAWINGAREA_H
#define DRAWINGAREA_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QMouseEvent>
#include <QResizeEvent>
#include <Eigen/Dense>

class DrawingArea : public QWidget {
    Q_OBJECT

private:
    QImage canvas;          
    QPoint lastPoint;       
    int penSize;            
    bool isDrawing;         

    void drawPoint(const QPoint& pos);
    void drawLine(const QPoint& from, const QPoint& to);
   
    Eigen::VectorXd imageToVector() const;
    Eigen::VectorXd preprocessImage() const;

public:
    explicit DrawingArea(QWidget* parent = nullptr);
    ~DrawingArea() = default;

    void clearCanvas();
    Eigen::VectorXd getProcessedImage() const;
    QImage getCanvas() const;

    void setPenSize(int size);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

signals:
    void drawingFinished();  
    void imageChanged();     
};

#endif 