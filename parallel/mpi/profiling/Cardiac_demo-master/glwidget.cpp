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

#include <QtGui>
#include <QtOpenGL>
#include <QThread>
#include <QImage>
#include <QPainter>
#include <QPixmap>
#include <QString>
#include <math.h>
#include <iostream>
#include "glwidget.h"

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif
#include <assert.h>
GLWidget::GLWidget(QWidget *parent, Mesh *_mesh)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    xRot = 0;
    yRot = 0;
    zRot = 0;

    saveImages = false;
    imcounter = 0;
    qtGreen = QColor::fromCmykF(0.40, 0.0, 1.0, 0.0);
    qtPurple = QColor::fromCmykF(0.39, 0.39, 0.0, 0.0);
    RangeMax = 40.0;
    RangeMin = -85.0;

    mesh = _mesh;
    mesh->normalizeForDrawing();
    mesh->build_facets();
    mesh->calcBoundaryFacetsNormals();

    vbuf1.resize(mesh->getPoints().size());
    vbuf2.resize(mesh->getPoints().size());

    memset(vbuf1.data(), 0, vbuf1.size()*sizeof(float));
    mpi_worker = new MPI_Worker(&vbuf1, &vbuf2, mesh, &mutex);
    QThread* thread = new QThread;
    connect(thread, SIGNAL(started()), mpi_worker, SLOT(run()), Qt::QueuedConnection);
    if(!connect(mpi_worker, SIGNAL(draw()), this, SLOT(redraw()))) {
        cout << "CONNECT ERROR" << endl;
    }
    connect(mpi_worker, SIGNAL(finished()), thread, SLOT(quit()));
    connect(mpi_worker, SIGNAL(finished()), this, SLOT(mpi_worker_finished()));

    isWireframeMode = true;
    isRotateMode = true;
    setFocusPolicy(Qt::StrongFocus);
    thread->start();
}
void GLWidget::keyPressEvent(QKeyEvent* e) {
    switch(e->key()) {
    case Qt::Key_W:
        isWireframeMode = !isWireframeMode;
        break;
    case Qt::Key_R:
        isRotateMode = !isRotateMode;
        break;

    default:
        break;
    }
}
GLWidget::~GLWidget()
{
}

QSize GLWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GLWidget::sizeHint() const
{
    return QSize(800, 800);
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void GLWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot) {
        xRot = angle;
        emit xRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
        emit yRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        emit zRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::redraw() {
    paintGL();
    saveFrame();
    QCoreApplication::processEvents();
}

void GLWidget::initializeGL()
{
    qglClearColor(Qt::GlobalColor::black);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_MULTISAMPLE);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);  // Nice perspective corrections
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Create light components
    GLfloat ambientLight[] = { 0.2f, 0.2f, 0.2f, 1.0f };
    GLfloat diffuseLight[] = { 0.3f, 0.3f, 0.3, 0.1f };
    GLfloat specularLight[] = { 0.5f, 0.5f, 0.5f, 1.0f };

    // GLfloat position[] = { -1, 0, 1, 0 };

    GLfloat position2[] = { 1, 0.5, 1, 0 };
    // Assign created components to GL_LIGHT0
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
    glLightfv(GL_LIGHT0, GL_POSITION, position2);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    float specReflection[] = { 0.5f, 0.5f, 0.5f, 1.0f };
    glMaterialfv(GL_FRONT, GL_SPECULAR, specReflection);
    GLfloat shininess[]={90.0f};
    // glMaterialfv(GL_FRONT, GL_SHININESS, shininess);

    GLdouble clipPlane[4] = {0,0,-1,-4};
    glClipPlane(GL_CLIP_PLANE0, clipPlane);

}

void GLWidget::paintGL()
{
    if (!mesh)
        return;

    if (!mutex.tryLock())
        return;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers
    glMatrixMode(GL_MODELVIEW);     // To operate on model-view matrix

    // Render a pyramid consists of 4 triangles
    glLoadIdentity();                  // Reset the model-view matrix
    glTranslatef(.0f, 0.0f, -4.0f);  // Move left and into the screen
    glRotatef(180,0.0,0.0,1.0);
    // glRotatef(-20,0.0,1.0,0.0);
    glRotatef(80,1.0,-0.4,0.0);

    if (isRotateMode) {
        zRot += 16;
    }
    // yRot += 8;
    // glRotatef(45,1.0,0.0,1.0);
    glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
    glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
    glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);

    auto facets = mesh->getBoundaryFacets();
    glPolygonMode( GL_FRONT_AND_BACK, isWireframeMode ? GL_LINE : GL_FILL );
    // glEnable(GL_CLIP_PLANE0);
    glBegin(GL_TRIANGLES);           // Begin drawing the pyramid with 4 triangles

    for (auto &f : facets){
        float *n = f.normal;
        glNormal3f(n[0],n[1],n[2]);
        for (int i=0; i<3; i++) {

            float *c = mesh->GetPointC(f.points[i]);

            setColor(vbuf1[f.points[i]], RangeMax, RangeMin);

            glVertex3f(c[0],c[1],c[2]);
        }
    }

    glEnd();   // Done drawing the pyramid



    glEnable(GL_LIGHTING);
    glFlush();
    swapBuffers();
    mutex.unlock();
}

void GLWidget::saveFrame() {
    if (saveImages) {
        // QImage img(this->size());
        // QPainter painter(&img);
        // render(&painter);
        stringstream ss;
        ss << imcounter++;
        QString filename("frames/img_"+QString::fromUtf8(ss.str().c_str())+".jpg");
        QPixmap image = QPixmap::grabWidget(this);
        if( !image.save( filename, "JPG" ) ){
            QMessageBox::warning( this, "Save Image", "Error saving image." );
        }
        // img.save("frames/img_"+ss.str()+".jpg");

    }
}

void GLWidget::perspective(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar)
{
    GLdouble xmin, xmax, ymin, ymax;

    ymax = zNear * tan( fovy * M_PI / 360.0 );
    ymin = -ymax;
    xmin = ymin * aspect;
    xmax = ymax * aspect;

    glFrustum( xmin, xmax, ymin, ymax, zNear, zFar );
}

void GLWidget::resizeGL(int width, int height)
{
    // Compute aspect ratio of the new window
    if (height == 0) height = 1;                // To prevent divide by 0
    GLfloat aspect = (GLfloat)width / (GLfloat)height;

    // Set the viewport to cover the new window

    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    perspective(45.0f, aspect, 0.1f, 100.0f);

}

void GLWidget::mousePressEvent(QMouseEvent *event)
{

    lastPos = event->pos();

}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();


    if (event->buttons() & Qt::LeftButton) {
        setXRotation(xRot + 8 * dy);
        setYRotation(yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(xRot + 8 * dy);
        setZRotation(zRot + 8 * dx);
    }
    lastPos = event->pos();

}

inline
void GLWidget::setColor(float v, float Max, float Min) {
    float range = Max-Min;
    float r;
    v -= Min;
    if (v < 0) r = 0;
    if (v > range) r = range;
    else r = v/range;
    int step = floor(r*4.0);
    r = (r-0.25*step)*4.0;
    float R = 0.f, G = 0.f, B = 0.f;
    switch(step){
    case 0:
        R = 0.f;
        G = r;
        B = 1.0f;
        break;
    case 1:
        R = 0.f;
        B = 1.0f-r;
        G = 1.f;
        break;
    case 2:
        G = 1.f;
        B = 0.f;
        R = r;
        break;
    case 3:
        B = 0.f;
        R = 1.f;
        G = 1.f-r;
        break;
    }
    glColor3f(R,G,B);
}

MPI_Worker::MPI_Worker(vector<float> *v1, vector<float> *v2,
                       Mesh *m, QMutex *mut){
    vbuf1 = v1;
    vbuf2 = v2;
    mesh = m;
    vector<int> rlens;
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert( rank == size - 1);
    int work_size = size - 1;
    int node_number = mesh->getPoints().size();

    for (int i=0; i<work_size; i++) {
        int r_len;
        r_len = node_number / work_size;
        if (i < (node_number % work_size)){
            r_len++;
        }
        rlens.push_back(r_len);
    }
    rlens.push_back(0);

    vizGathervRcounts = rlens;
    vizGathervDispls.resize(size);
    vizGathervDispls[0] = 0;
    for (int i=1; i<size; i++) {
        vizGathervDispls[i] = vizGathervDispls[i-1]+vizGathervRcounts[i-1];
    }
    mutex = mut;
    initialIdOrder.resize(node_number);
    MPI_Status st;
    MPI_Recv(initialIdOrder.data(), node_number, MPI_INT,
             0, 1234, MPI_COMM_WORLD, &st);
    // MPI_Irecv(NULL,0,MPI_INT,0,1111, MPI_COMM_WORLD, &finiReq);
    MPI_Irecv(&coll_count,1,MPI_INT,0,999,MPI_COMM_WORLD,&finiReq);
}

void MPI_Worker::run() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request req;
    MPI_Status st;
    int finiReqCompleted = 0;
    int completed = 1;
    int gstarted = 0;
    while(!finiReqCompleted) {
        if (completed) {
            MPI_Igatherv(NULL, 0, MPI_FLOAT,
                         vbuf2->data(), vizGathervRcounts.data(),
                         vizGathervDispls.data(),
                         MPI_FLOAT,size-1, MPI_COMM_WORLD, &req);
            gstarted++;
        }


        MPI_Test(&req, &completed, &st);
        if (completed) {
            mutex->lock();
            for (int i=0; i<vbuf2->size(); i++) {
                (*vbuf1)[initialIdOrder[i]] = (*vbuf2)[i];
            }
            mutex->unlock();

            emit draw();
        }
        MPI_Test(&finiReq,&finiReqCompleted, &st);
    }

    if (gstarted > coll_count)
        MPI_Cancel(&req);
    else {
        if (!completed)
            MPI_Wait(&req,&st);
        for (int i=gstarted; i<coll_count; i++) {
            MPI_Igatherv(NULL, 0, MPI_FLOAT,
                         vbuf2->data(), vizGathervRcounts.data(),
                         vizGathervDispls.data(),
                         MPI_FLOAT,size-1, MPI_COMM_WORLD, &req);
            MPI_Wait(&req,&st);
        }
    }
    emit finished();
}

void GLWidget::mpi_worker_finished() {
    delete mesh;
    mesh = NULL;
    QApplication::quit();
}
