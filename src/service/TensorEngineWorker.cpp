#include "TensorEngineWorker.h"

#include <QLoggingCategory>


namespace service
{
Q_LOGGING_CATEGORY(QLC_TENSOR_WORKER, "TensorEngineWorker")

using Tensor = engines::ITensorEngine::Tensor;

TensorEngineWorker::TensorEngineWorker(engines::ITensorEnginePtr const& engine, QObject* parent)
    : QObject(parent)
    , m_engine(engine)
{
}

TensorEngineWorker::~TensorEngineWorker()
{
    if (running())
    {
        stop();
    }
}

bool TensorEngineWorker::running() const
{
    return m_running;
}

int TensorEngineWorker::queueSize() const
{
    return m_requests.size();
}

size_t TensorEngineWorker::maxBatches() const
{
    return m_engine->maxBatches();
}

void TensorEngineWorker::start()
{
    if (running())
    {
        qCWarning(QLC_TENSOR_WORKER) << "Already started";
        return;
    }
    if (m_stop)
    {
        qCWarning(QLC_TENSOR_WORKER) << "Stop in proggress";
        return;
    }

    qCInfo(QLC_TENSOR_WORKER) << "Start requiered";

    m_stop = false;
    m_thread = std::thread([this] {run();});
}

void TensorEngineWorker::stop()
{
    if (!running())
    {
        qCWarning(QLC_TENSOR_WORKER) << "Not started";
        return;
    }
    if (m_stop)
    {
        qCWarning(QLC_TENSOR_WORKER) << "Stop already requiered";
        return;
    }

    qCInfo(QLC_TENSOR_WORKER) << "Stop requiered";

    m_stop = true;
    m_notifier.notify_one();
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

void TensorEngineWorker::push(quint64 id, common::IEngineInputDataPtr const& data)
{
    if (m_stop)
    {
        qCWarning(QLC_TENSOR_WORKER) << "Reject request by stop" << id;
        error(id, SkinCancerDetectorServiceSource::StopService);
    }
    else
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_requests.append({id, data});
        m_notifier.notify_one();
    }
}

void TensorEngineWorker::setRunning(bool running)
{
    if (m_running == running)
    {
        return;
    }

    qCInfo(QLC_TENSOR_WORKER) << "Set running state:" << running;

    m_running = running;
    emit runningChanged(m_running);
}

void TensorEngineWorker::run()
{
    setRunning(true);

    while (!(m_stop && m_requests.empty()))
    {
        QList<Request> processedData;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            while (m_requests.empty())
            {
                m_notifier.wait(lock);
            }

            if (m_requests.empty())
            {
                continue;
            }

            if (static_cast<size_t>(m_requests.size()) > m_engine->maxBatches())
            {
                processedData = m_requests.mid(0, m_engine->maxBatches());
                m_requests.erase(m_requests.begin(), m_requests.begin() + m_engine->maxBatches());
            }
            else
            {
                processedData = std::move(m_requests);
            }
        }

        QVector<Tensor> output(processedData.size() * m_engine->outputSize());
        if (!(loadData(processedData)
              && m_engine->infer(processedData.size())
              && m_engine->unloadOutput(processedData.size(), output.data())))
        {
            sendFailed(processedData);
            continue;
        }

        for (int b = 0; b < processedData.size(); ++b)
        {
            auto const pos = output[b * m_engine->outputSize() + 0];
            auto const neg = output[b * m_engine->outputSize() + 1];
            emit result(processedData[b].id, pos, neg);
        }
    }

    setRunning(false);
}

void TensorEngineWorker::sendFailed(QList<TensorEngineWorker::Request> const& data)
{
    for (auto const& request : data)
    {
        emit error(request.id, SkinCancerDetectorServiceSource::System);
    }
}

bool TensorEngineWorker::loadData(QList<TensorEngineWorker::Request> const& data)
{
    for (int b = 0; b < data.size(); ++b)
    {
        data[b].data->load(b, *m_engine);
    }

    return true;
}
}
