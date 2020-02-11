QT -= gui
QT += remoteobjects

CONFIG += c++17 console
CONFIG -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS

HEADERS += \
    src/engines/ImageConvertorSettings.h \
    src/engines/TensorEngineSettings.h \
    src/service/Service.h \
    src/service/ServiceSettings.h \
    src/service/SettingsReader.h \
    src/service/TensorEngineWorker.h \
    src/engines/ImageConvertor.h \
    src/engines/TensorEngine.h

SOURCES += \
    src/engines/ImageConvertorSettings.cpp \
    src/engines/TensorEngineSettings.cpp \
    src/service/Service.cpp \
    src/service/ServiceSettings.cpp \
    src/service/SettingsReader.cpp \
    src/service/TensorEngineWorker.cpp \
    src/engines/ImageConvertor.cpp \
    src/engines/TensorEngine.cpp \
    src/main.cpp

REPC_SOURCE += \
    src/service/SkinCancerDetectorService.rep

qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

unix:!macx: LIBS += -L$$(TENSOR_RT_ROOT)/lib/ -lnvinfer -lnvinfer_plugin -lnvonnxparser -lnvonnxparser_runtime -lnvparsers
unix:!macx: LIBS += -L$$(CUDADIR)/lib64 -lcudart -lcublas -lcudnn
unix:!macx: LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

INCLUDEPATH += src/
INCLUDEPATH += $$(TENSOR_RT_ROOT)/include
DEPENDPATH += $$(TENSOR_RT_ROOT)/include
INCLUDEPATH += $$(CUDADIR)/include
DEPENDPATH += $$(CUDADIR)/include
