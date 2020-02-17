#!/bin/bash

TENSOR_RT_LIB_DIR=/home/kusovsky/Libraries/TensorRT-6.0.1.5/lib/
CUDA_LIB_DIR=/usr/local/cuda/lib64/
QT_LIB_DIR=/home/kusovsky/Qt/5.14.1/gcc_64/lib/

export LD_LIBRARY_PATH
LD_LIBRARY_PATH=$TENSOR_RT_LIB_DIR:$CUDA_LIB_DIR:$QT_LIB_DIR:$LD_LIBRARY_PATH
./SkinCancerDetectorService
