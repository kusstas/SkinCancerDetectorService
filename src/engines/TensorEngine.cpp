#include "TensorEngine.h"

#include <QLoggingCategory>
#include <QElapsedTimer>
#include <QFile>


namespace engines
{
Q_LOGGING_CATEGORY(QLC_TENSORRT, "TensorRT")
Q_LOGGING_CATEGORY(QLC_TENSOR_ENGINE, "TensorEngine")

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

TensorEngine::TensorEngine(TensorEngineBuildSettings const& settings)
{
    if (!settings.valid())
    {
        auto const msg = "Invalid setting for create tensor engine";
        qCCritical(QLC_TENSOR_ENGINE) << msg << "- settings:" << settings;
        throw std::runtime_error(msg);
    }
    else
    {
        qCInfo(QLC_TENSOR_ENGINE) << "Settings -" << settings;
    }

    m_countTestsForEstimate = settings.countTestsForEstimate();

    deserialize(settings);
    if (!m_engine)
    {
        build(settings);
        if (m_engine)
        {
            serialize(settings);
        }
    }

    if (!m_engine)
    {
        auto const msg = "Creating cuda engine failed";
        qCCritical(QLC_TENSOR_ENGINE) << msg;
        throw std::runtime_error(msg);
    }

    m_executionContext.reset(m_engine->createExecutionContext());
    if(!m_executionContext)
    {
        auto const msg = "Creating execution context failed";
        qCCritical(QLC_TENSOR_ENGINE) << msg;
        throw std::runtime_error(msg);
    }

    m_inputDim = m_engine->getBindingDimensions(0);
    m_outputDim = m_engine->getBindingDimensions(1);
    m_batchInputN = getSize(m_inputDim);
    m_batchOutputN = getSize(m_outputDim);

    qCInfo(QLC_TENSOR_ENGINE) << "Network readed input:" << m_inputDim << "output:" << m_outputDim;

    auto const inputSize = m_batchInputN * maxBatches() * sizeof(Tensor);
    auto const outputSize = m_batchOutputN * maxBatches() * sizeof(Tensor);

    void* input = nullptr;
    void* output = nullptr;

    auto const cudaMallocError = "Cuda memory alloc failed";
    auto const cudaMallocErrorPost = "bytes required:";
    if (cudaMalloc(&input, inputSize) != ::cudaSuccess)
    {
        qCCritical(QLC_TENSOR_ENGINE) << cudaMallocError << cudaMallocErrorPost << inputSize;
        throw std::runtime_error(cudaMallocError);
    }
    m_input.reset(input);

    if (cudaMalloc(&output, outputSize) != ::cudaSuccess)
    {
        qCCritical(QLC_TENSOR_ENGINE) << cudaMallocError << cudaMallocErrorPost << inputSize;
        throw std::runtime_error(cudaMallocError);
    }
    m_output.reset(output);
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

bool TensorEngine::loadToInput(size_t batch, size_t offset, Tensor const* src, size_t n)
{
    if (src == nullptr)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Input data is nullptr";
        return false;
    }
    if (batch >= maxBatches())
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Batch argument" << QString::number(batch)
                                      << "should be less than max batches"
                                      << QString::number(maxBatches());
        return false;
    }

    auto const free = m_batchInputN - offset;
    if (free <= 0)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Offset argument" << QString::number(offset)
                                      << "should be less than size of batch"
                                      << QString::number(m_batchInputN);
        return false;
    }
    if (n > free)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Count argument" << QString::number(n)
                                      << "should be less than size of batch with offset"
                                      << QString::number(free);
        return false;
    }

    auto const input = static_cast<void*>(static_cast<Tensor*>(m_input.get()) + batch * m_batchInputN + offset);
    auto const count = n * sizeof(Tensor);

    bool const result = cudaMemcpy(input, src, count, cudaMemcpyHostToDevice) == ::cudaSuccess;
    qCDebug(QLC_TENSOR_ENGINE) << "Copy data to device, batch:" << batch
                               << "offset:" << offset
                               << "count:" << n
                               << (result ? "completed" : "failed");

    return result;
}

bool TensorEngine::unloadOutput(size_t batches, Tensor* dst)
{
    if (dst == nullptr)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Input data is nullptr";
        return false;
    }
    if (batches == 0 || batches > maxBatches())
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Batches argument" << QString::number(batches)
                                      << "should be geater than 0 and less(include) than max batches"
                                      << QString::number(maxBatches());
        return false;
    }

    auto const count = batches * m_batchOutputN * sizeof(Tensor);
    bool const result = cudaMemcpy(dst, m_output.get(), count, cudaMemcpyDeviceToHost)  == ::cudaSuccess;

    qCDebug(QLC_TENSOR_ENGINE) << "Unload data from device, batches:" << batches
                               << (result ? "completed" : "failed");

    return true;
}

bool TensorEngine::infer(size_t batches)
{
    if (batches > maxBatches())
    {
        return false;
    }

    qCInfo(QLC_TENSOR_ENGINE) << "Starting infer batches" << batches;

    void* bindings[] = {m_input.get(), m_output.get()};
    bool const result = m_executionContext->execute(batches, bindings);

    qCInfo(QLC_TENSOR_ENGINE) << "Infer batches" << batches << (result ? "completed" : "failed");

    return result;
}

qint64 TensorEngine::estimateInfer()
{
    qCInfo(QLC_TENSOR_ENGINE) << "Estimate infer starting";

    auto const estimateSuccess = [] (qint64 milliseconds) {
        qCInfo(QLC_TENSOR_ENGINE) << "Estimate infer completed:" << milliseconds << "nanoseconds";
        return milliseconds;
    };

    auto const estimateFailed = [] () {
        qCInfo(QLC_TENSOR_ENGINE) << "Estimate infer failed";
        return -1;
    };

    QElapsedTimer timer;
    std::unique_ptr<float[]> const dummyInput(new float[maxBatches() * m_batchInputN]);
    std::unique_ptr<float[]> const dummyOutput(new float[maxBatches() * m_batchOutputN]);

    timer.start();
    for (size_t i = 0; i < m_countTestsForEstimate; ++i)
    {
        for (size_t b = 0; b < maxBatches(); ++b)
        {
            if(!loadToInput(b, 0, dummyInput.get(), m_batchInputN))
            {
                return estimateFailed();
            }
        }

        if(!infer(maxBatches()))
        {
            return estimateFailed();
        }

        if(!unloadOutput(maxBatches(), dummyOutput.get()))
        {
            return estimateFailed();
        }
    }

    return estimateSuccess(timer.nsecsElapsed() / m_countTestsForEstimate);
}

void TensorEngine::serialize(TensorEngineBuildSettings const& settings)
{
    qCInfo(QLC_TENSOR_ENGINE) << "Trying serilize file" << settings.serializedFilePath();

    bool serialized = false;

    IHostMemoryPtr const hostMemory(m_engine->serialize());
    if (hostMemory)
    {
        QFile file(settings.serializedFilePath());
        if (file.open(QFile::WriteOnly))
        {
            auto const writen = file.write(static_cast<char const*>(hostMemory->data()), hostMemory->size());
            if (writen != static_cast<qint64>(hostMemory->size()))
            {
                qCCritical(QLC_TENSOR_ENGINE) << "Writing file" << settings.serializedFilePath()
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
            qCCritical(QLC_TENSOR_ENGINE) << "Opening file" << settings.serializedFilePath() << "for serilizing failed";
        }
    }
    else
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Creating host memory for serilizing failed";
    }

    if (serialized)
    {
        qCInfo(QLC_TENSOR_ENGINE) << "File" << settings.serializedFilePath() << "serialized";
    }
    else
    {
        qCWarning(QLC_TENSOR_ENGINE) << "Serializing file" << settings.serializedFilePath() << "failed";
    }
}

void TensorEngine::deserialize(TensorEngineBuildSettings const& settings)
{
    qCInfo(QLC_TENSOR_ENGINE) << "Trying deserilize file" << settings.serializedFilePath();

    if (QFile::exists(settings.serializedFilePath()))
    {
        IRuntimePtr const runtime(nvinfer1::createInferRuntime(gLogger));
        if (!runtime)
        {
            qCCritical(QLC_TENSOR_ENGINE) << "Creating infer runtime failed";
            return;
        }

        QFile file(settings.serializedFilePath());

        if (!file.open(QFile::ReadOnly))
        {
            qCCritical(QLC_TENSOR_ENGINE) << "Opening serialized file"
                                          << settings.serializedFilePath() << "failed";
            return;
        }

        auto const data = file.readAll();
        m_engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    }

    if (m_engine)
    {
        qCInfo(QLC_TENSOR_ENGINE) << "File" << settings.serializedFilePath() << "deserialized";
    }
    else
    {
        qCWarning(QLC_TENSOR_ENGINE) << "Deserializing file" << settings.serializedFilePath() << "failed";
    }
}

void TensorEngine::build(TensorEngineBuildSettings const& settings)
{
    qCInfo(QLC_TENSOR_ENGINE) << "Trying build model" << settings.onnxFilePath();

    IBuilderPtr const builder(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Creating infer builder failed";
        return;
    }

    auto const networkFlags = static_cast<nvinfer1::NetworkDefinitionCreationFlags>
                              (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinitionPtr const network(builder->createNetworkV2(networkFlags));
    if (!network)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Creating networkV2 failed";
        return;
    }

    IParserPtr const parser(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Creating onnx parser failed";
        return;
    }

    IBuilderConfigPtr const config(builder->createBuilderConfig());
    if (!config)
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Creating builder config failed";
        return;
    }

    if (!parser->parseFromFile(qPrintable(settings.onnxFilePath()),
                               static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)))
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Parsing onnx file" << settings.onnxFilePath() << "failed";
        return;
    }

    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setMaxWorkspaceSize(settings.maxWorkspaceSize());
    builder->setMaxBatchSize(settings.maxBatches());
    m_engine.reset(builder->buildEngineWithConfig(*network, *config));

    if (m_engine)
    {
        qCInfo(QLC_TENSOR_ENGINE) << "Model" << settings.onnxFilePath() << "builded";
    }
    else
    {
        qCCritical(QLC_TENSOR_ENGINE) << "Building engine failed";
    }
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
