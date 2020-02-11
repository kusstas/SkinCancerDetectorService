#pragma once

#include "TensorEngineSettings.h"

#include <memory>
#include <cstdint>
#include <QString>

#include <cuda.h>
#include <cuda_runtime_api.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#pragma GCC diagnostic pop


namespace engines
{
/**
 * @brief The TensorEngine class forward data through Neural Network
 */
class TensorEngine
{
public:
    /**
     * Tensor type
     */
    using Tensor = float;

    /**
     * @brief TensorEngine build engine by settings
     * @param settings should be valid otherwise throws runtime_exception
     */
    explicit TensorEngine(TensorEngineBuildSettings const& settings);

    /**
     * @brief max batches in TensorEngine(max batches which can forward engine)
     * @return size_t
     */
    size_t maxBatches() const;

    /**
     * @brief input width data
     * @return size_t
     */
    size_t inputWidth() const;

    /**
     * @brief input height data
     * @return size_t
     */
    size_t inputHeight() const;

    /**
     * @brief input channels data
     * @return size_t
     */
    size_t inputChannels() const;

    /**
     * @brief output size after forward
     * @return size_t
     */
    size_t outputSize() const;

    /**
     * @brief load to tnput data to device
     * @param batch - number of batch
     * @param offset - index of memory batch where will be record data
     * @param src - source data
     * @param n - count data need to load to device
     * @return bool - success
     */
    bool loadToInput(size_t batch, size_t offset, Tensor const* src, size_t n);

    /**
     * @brief unload output from device
     * @param batches - count batches to unload
     * @param dst - destination memory
     * @return bool - success
     */
    bool unloadOutput(size_t batches, Tensor* dst);

    /**
     * @brief forward data
     * @param batches - count batches for forward
     * @return bool - success
     */
    bool infer(size_t batches);

    /**
     * @brief forward dummy data for estimates
     * @return qint64 - milliseconds of forward
     */
    qint64 estimateInfer();

private:
    /**
     * @brief serialize builded ICudaEngine to file
     * @param settings - for get serializing file path
     */
    void serialize(TensorEngineBuildSettings const& settings);

    /**
     * @brief deserialize builded engine to memory
     * @param settings - for get serializing file path
     */
    void deserialize(TensorEngineBuildSettings const& settings);

    /**
     * @brief build engine from onnx file
     * @param settings build engine
     */
    void build(TensorEngineBuildSettings const& settings);

    /**
     * @brief get size of dimension
     * @param dims
     * @return size
     */
    static size_t getSize(nvinfer1::Dims const& dims);

private:
    /**
     * @brief The NvDeleter struct - custom deleter for tensotRT interfaces
     */
    template <typename T>
    struct NvDeleter
    {
        void operator () (T* obj)
        {
            obj->destroy();
        }
    };

    /**
     * @brief The CudaDeleter struct - custom deleter for cuda memory
     */
    struct CudaDeleter
    {
        void operator () (void* add)
        {
            cudaFree(add);
        }
    };

    // Smart pointers with overrided deleters
    using ICudaEnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, NvDeleter<nvinfer1::ICudaEngine>>;
    using IExecutionContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, NvDeleter<nvinfer1::IExecutionContext>>;
    using IRuntimePtr = std::unique_ptr<nvinfer1::IRuntime, NvDeleter<nvinfer1::IRuntime>>;
    using IBuilderPtr = std::unique_ptr<nvinfer1::IBuilder, NvDeleter<nvinfer1::IBuilder>>;
    using IBuilderConfigPtr = std::unique_ptr<nvinfer1::IBuilderConfig, NvDeleter<nvinfer1::IBuilderConfig>>;
    using INetworkDefinitionPtr = std::unique_ptr<nvinfer1::INetworkDefinition, NvDeleter<nvinfer1::INetworkDefinition>>;
    using IHostMemoryPtr = std::unique_ptr<nvinfer1::IHostMemory, NvDeleter<nvinfer1::IHostMemory>>;
    using IParserPtr = std::unique_ptr<nvonnxparser::IParser, NvDeleter<nvonnxparser::IParser>>;
    using CudaMemPtr = std::unique_ptr<void, CudaDeleter>;

private:
    CudaMemPtr m_input = nullptr;
    CudaMemPtr m_output = nullptr;
    nvinfer1::Dims m_inputDim{};
    nvinfer1::Dims m_outputDim{};
    size_t m_batchInputN = 0;
    size_t m_batchOutputN = 0;

    ICudaEnginePtr m_engine = nullptr;
    IExecutionContextPtr m_executionContext = nullptr;
};
}
