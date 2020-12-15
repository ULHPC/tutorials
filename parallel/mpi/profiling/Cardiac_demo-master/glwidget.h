/****************************************************************************
 **
 ** Copyright (C) 2014 Digia Plc and/or its subsidiary(-ies).
 ** Contact: http://www.qt-project.org/legal
 **
 ** This file is part of the examples of the Qt Toolkit.
 **
 ** $QT_BEGIN_LICENSE:BSD$
 ** You may use this file under the terms of the BSD license as follows:
 **
 ** "Redistribution and use in source and binary forms, with or without
 ** modification, are permitted provided that the following conditions are
 ** met:
 **   * Redistributions of source code must retain the above copyright
 **     notice, this list of conditions and the following disclaimer.
 **   * Redistributions in binary form must reproduce the above copyright
 **     notice, this list of conditions and the following disclaimer in
 **     the documentation and/or other materials provided with the
 **     distribution.
 **   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
 **     of its contributors may be used to endorse or promote products derived
 **     from this software without specific prior written permission.
 **
 **
 ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 ** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 ** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 ** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 ** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 ** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 ** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 **
 ** $QT_END_LICENSE$
 **
 ****************************************************************************/

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include "mesh.hpp"
#include <mpi.h>
#include <QMutex>

class MPI_Worker : public QObject {
    Q_OBJECT
private:
    vector<float> *vbuf1;
    vector<float> *vbuf2;
    Mesh *mesh;
    vector<int> vizGathervRcounts;
    vector<int> vizGathervDispls;
    QMutex *mutex;
    vector<int> initialIdOrder;
    MPI_Request finiReq;
    int coll_count;
public:
    MPI_Worker(vector<float> *v1, vector<float> *v2,
               Mesh *m, QMutex *mut);

    public slots:
        void run();
signals:
        void draw();
        void finished();
};



class GLWidget : public QGLWidget
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent, Mesh *_mesh);
    ~GLWidget();

    void saveFrame();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    void set_mesh(Mesh *_mesh);
    void resize(int width, int height) { resizeGL(width,height); }
    void setRange(float max, float min) {
        RangeMin = min;
        RangeMax = max;
    }
    void SetSaveImages(bool v) {
        saveImages = v;
    }
    public slots:
        void setXRotation(int angle);
        void setYRotation(int angle);
        void setZRotation(int angle);
        void redraw();
        void mpi_worker_finished();
signals:
        void xRotationChanged(int angle);
        void yRotationChanged(int angle);
        void zRotationChanged(int angle);




protected:
        void initializeGL();
        void paintGL();
        void resizeGL(int width, int height);
        void mousePressEvent(QMouseEvent *event);
        void mouseMoveEvent(QMouseEvent *event);
        void perspective(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar);
        void setColor(float v, float Max, float Min);
        virtual void  keyPressEvent(QKeyEvent *event);
private:
        bool isWireframeMode;
        bool isRotateMode;
        bool saveImages;
        int imcounter;
        MPI_Worker *mpi_worker;
        int xRot;
        int yRot;
        int zRot;
        float RangeMax;
        float RangeMin;
        vector<float> vbuf1;
        vector<float> vbuf2;
        QPoint lastPos;
        QColor qtGreen;
        QColor qtPurple;
        Mesh *mesh;
        QMutex mutex;
};

#endif
