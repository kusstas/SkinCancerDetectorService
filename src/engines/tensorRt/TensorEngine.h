#pragma once

#include "engines/BaseTensorEngine.h"

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
class BaseTensorEngineSettings;

namespace tensorRt
{
class TensorEngineSettings;

/**
 * @brief The TensorEngine class forward data through Neural Network by TensorRt Framework
 */
class TensorEngine : public engines::BaseTensorEngine
{
public:
    TensorEngine() = default;

public: // BaseTensorEngine interface
    bool loadImpl(BaseTensorEngineSettings const& settings) override;

public: // ITensorEngine interface
    size_t maxBatches() const override;
    size_t inputWidth() const override;
    size_t inputHeight() const override;
    size_t inputChannels() const override;
    size_t outputSize() const override;
    size_t batchInputN() const override;
    size_t batchOutputN() const override;
    bool loadToInput(size_t batch, size_t offset, Tensor const* src, size_t n) override;
    bool unloadOutput(size_t batches, Tensor* dst) override;
    bool infer(size_t batches) override;

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
    /**
     * @brief serialize builded ICudaEngine to file
     * @param engine for serialize
     * @param settings - for get serializing file path
     */
    void serialize(nvinfer1::ICudaEngine* engine, TensorEngineSettings const& settings);

    /**
     * @brief deserialize builded engine to memory
     * @param settings - for get serializing file path
     */
    ICudaEnginePtr deserialize(TensorEngineSettings const& settings);

    /**
     * @brief build engine from onnx file
     * @param settings build engine
     */
    ICudaEnginePtr build(TensorEngineSettings const& settings);

    /**
     * @brief get size of dimension
     * @param dims
     * @return size
     */
    static size_t getSize(nvinfer1::Dims const& dims);

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
}
