#include "TensorEngine.h"

#include <QLoggingCategory>


namespace engines
{
namespace torch
{
Q_LOGGING_CATEGORY(QLC_TORCH, "TorchEngine")

bool TensorEngine::loadImpl(BaseTensorEngineSettings const& settings)
{
    auto const& torchSettigs = settings.toInstance<torch::TensorEngineSettings>();

    if (!torchSettigs)
    {
        qCCritical(QLC_TORCH) << "Invalid setting for load torch tensor engine";
        return false;
    }
    m_settings = torchSettigs;
    m_batchInputN = inputWidth() * inputHeight() * inputChannels();

    if (!m_settings->device().isEmpty())
    {
        try
        {
            m_device = c10::Device(qPrintable(m_settings->device()));
        }
        catch (std::exception const& ex)
        {
            qCCritical(QLC_TORCH) << "Cannot create device, reason:" << ex.what();
            return false;
        }
    }

    try
    {
        m_module = ::torch::jit::load(qPrintable(m_settings->modelPath()), m_device);
    }
    catch (std::exception const& ex)
    {
        qCCritical(QLC_TORCH) << "Cannot load script:" << m_settings->modelPath() << "reason:" << ex.what();
        return false;
    }

    m_input = std::vector<Tensor>(batchInputN() * maxBatches());

    qCInfo(QLC_TORCH) << "Model" << m_settings->modelPath() << "loaded";
    return true;
}

size_t TensorEngine::maxBatches() const
{
    return m_settings->maxBatches();
}

size_t TensorEngine::inputWidth() const
{
    return m_settings->width();
}

size_t TensorEngine::inputHeight() const
{
    return m_settings->height();
}

size_t TensorEngine::inputChannels() const
{
    return m_settings->channels();
}

size_t TensorEngine::outputSize() const
{
    return m_settings->output();
}

bool TensorEngine::loadToInput(size_t batch, size_t offset, Tensor const* src, size_t n)
{
    if (!validateLoadInput(batch, offset, src, n))
    {
        return false;
    }

    auto const input = m_input.data() + batch * batchInputN() + offset;
    std::copy(src, src + n, input);

    return true;
}

bool TensorEngine::unloadOutput(size_t batches, Tensor* dst)
{
    if (!validateLoadOutput(batches, dst))
    {
        return false;
    }

    auto const tensor = m_output.toTensor().cpu();
    auto const n = batches * batchOutputN();
    auto const src = tensor.data_ptr<float>();

    if (tensor.dim() < 1)
    {
        return false;
    }

    std::copy(src, src + n, dst);

    return true;
}

bool TensorEngine::infer(size_t batches)
{
    if (!validateInfer(batches))
    {
        return false;
    }

    auto ivalue = ::torch::from_blob(
                m_input.data(),
    {static_cast<int>(batches),
     static_cast<int>(inputChannels()),
     static_cast<int>(inputHeight()),
     static_cast<int>(inputWidth())},
                ::torch::kFloat32);

    if (m_device)
    {
        ivalue = ivalue.to(*m_device);
    }

    try
    {
        qCInfo(QLC_TORCH) << "Starting infer batches" << batches;
        m_output = m_module.forward({ivalue});
    }
    catch(std::exception const& ex)
    {
        qCCritical(QLC_TORCH) << "Forward failed, reason:" << ex.what();
        return false;
    }

    qCInfo(QLC_TORCH) << "Infer batches" << batches << "completed";

    return true;
}

size_t TensorEngine::batchInputN() const
{
    return m_batchInputN;
}

size_t TensorEngine::batchOutputN() const
{
    return outputSize();
}
}
}
