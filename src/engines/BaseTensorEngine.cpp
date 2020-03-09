#include "BaseTensorEngine.h"

#include <QLoggingCategory>
#include <QElapsedTimer>


namespace engines
{
Q_LOGGING_CATEGORY(QLC_BASE_TENSOR_ENGINE, "BaseTensorEngine")

BaseTensorEngineSettings const& BaseTensorEngine::settings() const
{
    return m_settings;
}

qint64 BaseTensorEngine::estimate()
{
    qCInfo(QLC_BASE_TENSOR_ENGINE) << "Estimate infer starting";

    auto const estimateSuccess = [] (qint64 milliseconds) {
        qCInfo(QLC_BASE_TENSOR_ENGINE) << "Estimate infer completed:" << milliseconds << "nanoseconds";
        return milliseconds;
    };

    auto const estimateFailed = [] () {
        qCInfo(QLC_BASE_TENSOR_ENGINE) << "Estimate infer failed";
        return -1;
    };

    QElapsedTimer timer;
    std::vector<float> const dummyInput(maxBatches() * batchInputN());
    std::vector<float> dummyOutput(maxBatches() * batchOutputN());

    timer.start();
    for (size_t i = 0; i < settings().countTestsForEstimate(); ++i)
    {
        for (size_t b = 0; b < maxBatches(); ++b)
        {
            if(!loadToInput(b, 0, dummyInput.data(), batchInputN()))
            {
                return estimateFailed();
            }
        }

        if(!infer(maxBatches()))
        {
            return estimateFailed();
        }

        if(!unloadOutput(maxBatches(), dummyOutput.data()))
        {
            return estimateFailed();
        }
    }

    return estimateSuccess(timer.nsecsElapsed() / settings().countTestsForEstimate());
}

bool BaseTensorEngine::load(BaseTensorEngineSettings const& settings)
{
    if (!settings.valid())
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Invalid setting for load tensor engine";
        return false;
    }

    m_settings = settings;

    loadImpl(settings);

    if (m_settings.positiveIndex() >= outputSize() || m_settings.negativeIndex() >= outputSize())
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Invalid indexes for output tensor engine";
        return false;
    }

    return true;
}

size_t BaseTensorEngine::positiveIndex() const
{
    return settings().positiveIndex();
}

size_t BaseTensorEngine::negativeIndex() const
{
    return settings().negativeIndex();
}

bool BaseTensorEngine::loadImpl(BaseTensorEngineSettings const&)
{
    return false;
}

bool BaseTensorEngine::validateLoadInput(size_t batch, size_t offset, Tensor const* src, size_t n) const
{
    if (src == nullptr)
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Input data is nullptr";
        return false;
    }
    if (batch >= maxBatches())
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Batch argument" << QString::number(batch)
                                           << "should be less than max batches"
                                           << QString::number(maxBatches());
        return false;
    }

    auto const free = batchInputN() - offset;
    if (free <= 0)
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Offset argument" << QString::number(offset)
                                           << "should be less than size of batch"
                                           << QString::number(batchInputN());
        return false;
    }
    if (n > free)
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Count argument" << QString::number(n)
                                           << "should be less than size of batch with offset"
                                           << QString::number(free);
        return false;
    }

    return true;
}

bool BaseTensorEngine::validateLoadOutput(size_t batches, Tensor* dst) const
{
    if (dst == nullptr)
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Input data is nullptr";
        return false;
    }
    if (batches == 0 || batches > maxBatches())
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Batches argument" << QString::number(batches)
                                           << "should be geater than 0 and less(include) than max batches"
                                           << QString::number(maxBatches());
        return false;
    }

    return true;
}

bool BaseTensorEngine::validateInfer(size_t batches) const
{
    if (batches > maxBatches())
    {
        qCCritical(QLC_BASE_TENSOR_ENGINE) << "Requered infer batches:" << batches
                                           << "should in range max batches:" << maxBatches();
        return false;
    }

    return true;
}
}
