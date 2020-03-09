#include "TensorEngine.h"
#include "TensorEngineSettings.h"

#include <QLoggingCategory>
#include <QFile>


namespace engines
{
namespace tensorRt
{
Q_LOGGING_CATEGORY(QLC_TENSORRT, "TensorRT")
Q_LOGGING_CATEGORY(QLC_TENSOR_RT_ENGINE, "TensorRtEngine")

QDebug operator<<(QDebug d, nvinfer1::Dims const& dims);

/**
 * @brief The Logger class - tensor engine logger
 */
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) override
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            qCCritical(QLC_TENSORRT) << msg;
            break;
        case Severity::kWARNING:
            qCWarning(QLC_TENSORRT) << msg;
            break;
        case Severity::kINFO:
            qCInfo(QLC_TENSORRT) << msg;
            break;
        case Severity::kVERBOSE:
            qCDebug(QLC_TENSORRT) << msg;
            break;
        }
    }
} gLogger;

bool TensorEngine::loadImpl(engines::BaseTensorEngineSettings const& settings)
{
    auto const& tensorRtSettigs = settings.toInstance<tensorRt::TensorEngineSettings>();
    if (!tensorRtSettigs)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Invalid setting for load tensor rt engine";
        return false;
    }

    auto engine = deserialize(*tensorRtSettigs);
    if (!engine)
    {
        engine = build(*tensorRtSettigs);
        if (engine)
        {
            serialize(engine.get(), *tensorRtSettigs);
        }
    }

    if (!engine)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating cuda engine failed";
        return false;
    }

    IExecutionContextPtr ctx(engine->createExecutionContext());

    if(!ctx)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating execution context failed";
        return false;
    }

    auto const inputDim = engine->getBindingDimensions(0);
    auto const outputDim = engine->getBindingDimensions(1);
    auto const batchInputN = getSize(inputDim);
    auto const batchOutputN = getSize(outputDim);

    qCInfo(QLC_TENSOR_RT_ENGINE) << "Network readed input:" << inputDim << "output:" << outputDim;

    auto const inputSize = batchInputN * engine->getMaxBatchSize() * sizeof(Tensor);
    auto const outputSize = batchOutputN * engine->getMaxBatchSize() * sizeof(Tensor);

    void* input = nullptr;
    void* output = nullptr;

    if (cudaMalloc(&input, inputSize) != ::cudaSuccess)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Cuda memory alloc failed bytes required:" << inputSize;
        return false;
    }
    CudaMemPtr inputPtr(input);

    if (cudaMalloc(&output, outputSize) != ::cudaSuccess)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Cuda memory alloc failed bytes required:" << outputSize;
        return false;
    }
    CudaMemPtr outputPtr(output);

    m_input = std::move(inputPtr);
    m_output = std::move(outputPtr);
    m_inputDim = inputDim;
    m_outputDim = outputDim;
    m_batchInputN = batchInputN;
    m_batchOutputN = batchOutputN;
    m_engine = std::move(engine);
    m_executionContext = std::move(ctx);

    return true;
}

size_t TensorEngine::maxBatches() const
{
    return m_engine->getMaxBatchSize();
}

size_t TensorEngine::inputWidth() const
{
    return m_inputDim.d[2];
}

size_t TensorEngine::inputHeight() const
{
    return m_inputDim.d[1];
}

size_t TensorEngine::inputChannels() const
{
    return m_inputDim.d[0];
}

size_t TensorEngine::outputSize() const
{
    return m_outputDim.d[0];
}

size_t TensorEngine::batchInputN() const
{
    return m_batchInputN;
}

size_t TensorEngine::batchOutputN() const
{
    return m_batchOutputN;
}

bool TensorEngine::loadToInput(size_t batch, size_t offset, Tensor const* src, size_t n)
{
    if (!validateLoadInput(batch, offset, src, n))
    {
        return false;
    }

    auto const input = static_cast<void*>(static_cast<Tensor*>(m_input.get()) + batch * batchInputN() + offset);
    auto const count = n * sizeof(Tensor);

    bool const result = cudaMemcpy(input, src, count, cudaMemcpyHostToDevice) == ::cudaSuccess;
    qCDebug(QLC_TENSOR_RT_ENGINE) << "Copy data to device, batch:" << batch
                                  << "offset:" << offset
                                  << "count:" << n
                                  << (result ? "completed" : "failed");

    return result;
}

bool TensorEngine::unloadOutput(size_t batches, Tensor* dst)
{
    if (!validateLoadOutput(batches, dst))
    {
        return false;
    }

    auto const count = batches * batchOutputN() * sizeof(Tensor);
    bool const result = cudaMemcpy(dst, m_output.get(), count, cudaMemcpyDeviceToHost)  == ::cudaSuccess;

    qCDebug(QLC_TENSOR_RT_ENGINE) << "Unload data from device, batches:" << batches
                                  << (result ? "completed" : "failed");

    return true;
}

bool TensorEngine::infer(size_t batches)
{
    if (!validateInfer(batches))
    {
        return false;
    }

    qCInfo(QLC_TENSOR_RT_ENGINE) << "Starting infer batches" << batches;

    void* bindings[] = {m_input.get(), m_output.get()};
    bool const result = m_executionContext->execute(batches, bindings);

    qCInfo(QLC_TENSOR_RT_ENGINE) << "Infer batches" << batches << (result ? "completed" : "failed");

    return result;
}

void TensorEngine::serialize(nvinfer1::ICudaEngine* engine, TensorEngineSettings const& settings)
{
    qCInfo(QLC_TENSOR_RT_ENGINE) << "Trying serilize file" << settings.serializedFilePath();

    bool serialized = false;

    IHostMemoryPtr const hostMemory(engine->serialize());
    if (hostMemory)
    {
        QFile file(settings.serializedFilePath());
        if (file.open(QFile::WriteOnly))
        {
            auto const writen = file.write(static_cast<char const*>(hostMemory->data()), hostMemory->size());
            if (writen != static_cast<qint64>(hostMemory->size()))
            {
                qCCritical(QLC_TENSOR_RT_ENGINE) << "Writing file" << settings.serializedFilePath()
                                                 << "for serilizing failed, "
                                                    "writen" << writen
                                                 << "neccessary" << hostMemory->size();
            }
            else
            {
                serialized = true;
            }
        }
        else
        {
            qCCritical(QLC_TENSOR_RT_ENGINE) << "Opening file" << settings.serializedFilePath() << "for serilizing failed";
        }
    }
    else
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating host memory for serilizing failed";
    }

    if (serialized)
    {
        qCInfo(QLC_TENSOR_RT_ENGINE) << "File" << settings.serializedFilePath() << "serialized";
    }
    else
    {
        qCWarning(QLC_TENSOR_RT_ENGINE) << "Serializing file" << settings.serializedFilePath() << "failed";
    }
}

TensorEngine::ICudaEnginePtr TensorEngine::deserialize(TensorEngineSettings const& settings)
{
    qCInfo(QLC_TENSOR_RT_ENGINE) << "Trying deserilize file" << settings.serializedFilePath();
    ICudaEnginePtr engine = nullptr;

    if (QFile::exists(settings.serializedFilePath()))
    {
        IRuntimePtr const runtime(nvinfer1::createInferRuntime(gLogger));
        if (!runtime)
        {
            qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating infer runtime failed";
            return engine;
        }

        QFile file(settings.serializedFilePath());

        if (!file.open(QFile::ReadOnly))
        {
            qCCritical(QLC_TENSOR_RT_ENGINE) << "Opening serialized file"
                                             << settings.serializedFilePath() << "failed";
            return engine;
        }

        auto const data = file.readAll();
        engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    }

    if (engine)
    {
        qCInfo(QLC_TENSOR_RT_ENGINE) << "File" << settings.serializedFilePath() << "deserialized";
    }
    else
    {
        qCWarning(QLC_TENSOR_RT_ENGINE) << "Deserializing file" << settings.serializedFilePath() << "failed";
    }

    return engine;
}

TensorEngine::ICudaEnginePtr TensorEngine::build(TensorEngineSettings const& settings)
{
    qCInfo(QLC_TENSOR_RT_ENGINE) << "Trying build model" << settings.onnxFilePath();
    ICudaEnginePtr engine = nullptr;

    IBuilderPtr const builder(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating infer builder failed";
        return engine;
    }

    auto const networkFlags = static_cast<nvinfer1::NetworkDefinitionCreationFlags>
            (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinitionPtr const network(builder->createNetworkV2(networkFlags));
    if (!network)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating networkV2 failed";
        return engine;
    }

    IParserPtr const parser(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating onnx parser failed";
        return engine;
    }

    IBuilderConfigPtr const config(builder->createBuilderConfig());
    if (!config)
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Creating builder config failed";
        return engine;
    }

    if (!parser->parseFromFile(qPrintable(settings.onnxFilePath()),
                               static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)))
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Parsing onnx file" << settings.onnxFilePath() << "failed";
        return engine;
    }

    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setMaxWorkspaceSize(settings.maxWorkspaceSize());
    builder->setMaxBatchSize(settings.maxBatches());
    engine.reset(builder->buildEngineWithConfig(*network, *config));

    if (engine)
    {
        qCInfo(QLC_TENSOR_RT_ENGINE) << "Model" << settings.onnxFilePath() << "builded";
    }
    else
    {
        qCCritical(QLC_TENSOR_RT_ENGINE) << "Building engine failed";
    }

    return engine;
}

size_t TensorEngine::getSize(nvinfer1::Dims const& dims)
{
    size_t size = dims.nbDims > 0 ? 1 : 0;

    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }

    return size;
}

QDebug operator<<(QDebug d, nvinfer1::Dims const& dims)
{
    d.nospace()<< "{";
    if (dims.nbDims > 0)
    {
        d << dims.d[0];
        for (int i = 1; i < dims.nbDims; ++i)
        {
            d << ", " << dims.d[i];
        }
    }
    d << "}";

    return d.space();
}
}
}
