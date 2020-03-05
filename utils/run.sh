#!/bin/bash

function printHelp {
cat << EOF
usage:
    --tensor-engine: tensor engine type (torch|tensorRt), if not specified will be setup from json
    --qtlib: path to qt lib folder
    --opencvlib: path to opencv lib folder
    --cudalib: path to cuda lib folder
    --tensorrtlib: path to tensorrt lib filder
    --torchlib: path to torch lib folder
    --help: print usage
EOF
}

for i in "$@"
do
case $i in
    --tensor-engine=*)
    TENSOR_ENGINE="${i#*=}"
    shift
    ;;
    --qtlib=*)
    QT_LIB_DIR="${i#*=}"
    shift
    ;;
    --opencvlib=*)
    OPENCV_LIB_DIR="${i#*=}"
    shift
    ;;
    --cudalib=*)
    CUDA_LIB_DIR="${i#*=}"
    shift
    ;;
    --tensorrtlib=*)
    TENSOR_RT_LIB_DIR="${i#*=}"
    shift
    ;;
    --torchlib=*)
    TORCH_LIB_DIR="${i#*=}"
    shift
    ;;
    --help)
    printHelp
    exit 0
    ;;
    *)
        echo "unknown argument";
        printHelp
        exit 1
    ;;
esac
done

export LD_LIBRARY_PATH

if [ -z "$QT_LIB_DIR" ]; then
echo "--qtlib is not set!"
exit 1
fi


function checkTensorRt {
    if [ -z "$TENSOR_RT_LIB_DIR" ]; then echo "--tensorrtlib is not set!"; exit 1; fi;
    if [ -z "$CUDA_LIB_DIR" ]; then echo "--cudalib is not set!"; exit 1; fi;
}

function checkTorch {
    if [ -z "$TORCH_LIB_DIR" ]; then echo "--torchlib is not set!"; exit 1; fi;
}

if [ $TENSOR_ENGINE == "tensorRt" ]
then
checkTensorRt
elif [ $TENSOR_ENGINE == "torch" ]
then
checkTorch
else
checkTensorRt
checkTorch
fi

if [ -n "$OPENCV_LIB_DIR" ]; then LD_LIBRARY_PATH=$OPENCV_LIB_DIR:$LD_LIBRARY_PATH; fi
if [ -n "$CUDA_LIB_DIR" ]; then LD_LIBRARY_PATH=$CUDA_LIB_DIR:$LD_LIBRARY_PATH; fi;
if [ -n "$TENSOR_RT_LIB_DIR" ]; then LD_LIBRARY_PATH=$TENSOR_RT_LIB_DIR:$LD_LIBRARY_PATH; fi;
if [ -n "$TORCH_LIB_DIR" ]; then LD_LIBRARY_PATH=$TORCH_LIB_DIR:$LD_LIBRARY_PATH; fi;
LD_LIBRARY_PATH=$QT_LIB_DIR:$OPENCV_LIB_DIR:$LD_LIBRARY_PATH

./SkinCancerDetectorService --tensor-engine=$TENSOR_ENGINE
