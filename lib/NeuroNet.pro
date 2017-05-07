#-------------------------------------------------
#
# Project created by QtCreator 2017-05-05T17:09:52
#
#-------------------------------------------------

TARGET = NeuroNet
TEMPLATE = lib

DEFINES += NEURONET_LIBRARY

SOURCES += \
    tneuronet.cpp \
    tgeneticalgorithm.cpp \
    tdatasource.cpp

HEADERS += \
    tneuronet.h \
    tgeneticalgorithm.h \
    tdatasource.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}
