VPATH += ../shared
INCLUDEPATH += ../shared

HEADERS       = glwidget.h \
window.h \
mesh.hpp


SOURCES       = glwidget.cpp \
heart_demo.cpp \
window.cpp \
luo_rudy_1991.cpp \
mesh.cpp

QT           += opengl

QMAKE_CXX=mpiicpc
QMAKE_LINK=mpiicpc


QMAKE_CXXFLAGS="-g -std=c++11 -fopenmp ${HEART_EXTRA}"
QMAKE_CXXFLAGS+="-O3"
;; CONFIG+=debug
;; QMAKE_CXXFLAGS+="-O0"

QMAKE_LFLAGS="-fopenmp ${HEART_EXTRA} -lrcm"
