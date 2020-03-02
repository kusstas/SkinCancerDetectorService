QT -= gui
QT += remoteobjects

CONFIG += c++17 console
CONFIG += file_copies
CONFIG += object_parallel_to_source tensorrt
CONFIG -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS

HEADERS += \
    src/common/IEngineInputData.h \
    src/common/IEstimated.h \
    src/common/IJsonParsed.h \
    src/common/ISettings.h \
    src/engines/BaseTensorEngineSettings.h \
    src/engines/ITensorEngine.h \
    src/image/IImageConvertor.h \
    src/image/ImageConvertorSettings.h \
    src/image/opencv/ImageConvertor.h \
    src/service/ImageConvertorWorker.h \
    src/service/Service.h \
    src/service/ServiceSettings.h \
    src/service/TensorEngineWorker.h \
    src/utils/JsonHelper.h \
    src/utils/ServiceLocator.h \
    src/utils/SettingsReader.h

SOURCES += \
    src/engines/BaseTensorEngineSettings.cpp \
    src/image/ImageConvertorSettings.cpp \
    src/image/opencv/ImageConvertor.cpp \
    src/main.cpp \
    src/service/ImageConvertorWorker.cpp \
    src/service/Service.cpp \
    src/service/ServiceSettings.cpp \
    src/service/TensorEngineWorker.cpp \
    src/utils/ServiceLocator.cpp \
    src/utils/SettingsReader.cpp

tensorrt {
DEFINES += INCLUDE_TENSOR_RT_BUILD

HEADERS += src/engines/tensorRt/TensorEngine.h \
    src/engines/tensorRt/TensorEngineSettings.h \

SOURCES += \
    src/engines/tensorRt/TensorEngine.cpp \
    src/engines/tensorRt/TensorEngineSettings.cpp \
}

REPC_SOURCE += \
    src/service/SkinCancerDetectorService.rep


COPIES += utils_files

utils_files.files = $$PWD/utils/run.sh $$files($$PWD/resources/*)
utils_files.path = $$OUT_PWD

qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

unix:!macx: LIBS += -L$$(TENSOR_RT_ROOT)/lib/ -lnvinfer -lnvinfer_plugin -lnvonnxparser -lnvonnxparser_runtime -lnvparsers
unix:!macx: LIBS += -L$$(CUDADIR)/lib64 -lcudart -lcublas -lcudnn
LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

INCLUDEPATH += src/
INCLUDEPATH += $$(TENSOR_RT_ROOT)/include
DEPENDPATH += $$(TENSOR_RT_ROOT)/include
INCLUDEPATH += $$(CUDADIR)/include
DEPENDPATH += $$(CUDADIR)/include
INCLUDEPATH += $$(OPENCV_INCLUDE)
DEPENDPATH += $$(OPENCV_INCLUDE)
