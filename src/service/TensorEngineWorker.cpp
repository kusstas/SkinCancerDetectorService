#include "TensorEngineWorker.h"


namespace service
{
TensorEngineWorker::TensorEngineWorker(std::shared_ptr<engines::TensorEngine> const& engine, QObject* parent)
    : QObject(parent)
    , m_engine(engine)
{
}

bool TensorEngineWorker::running() const
{
    return m_running;
}

int TensorEngineWorker::queueSize() const
{
    return m_requests.size();
}

void TensorEngineWorker::start()
{
    m_stop = false;
    m_thread = std::thread([this] {run();});
}

void TensorEngineWorker::stop()
{
    m_stop = true;
    m_notifier.notify_one();
    m_thread.join();
}

void TensorEngineWorker::push(quint64 id, QVector<cv::Mat> const& data)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_requests.append({id, data});
    m_notifier.notify_one();
}

void TensorEngineWorker::setRunning(bool running)
{
    if (m_running == running)
    {
        return;
    }

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
              || m_engine->infer(processedData.size())
              || m_engine->unloadOutput(processedData.size(), output.data())))
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
        emit error(request.id);
    }
}

bool TensorEngineWorker::loadData(QList<TensorEngineWorker::Request> const& data)
{
    for (int b = 0; b < data.size(); ++b)
    {
        size_t offset = 0;
        auto const& channels = data[b].data;
        for (int ch = 0; ch < channels.size(); ++ch)
        {
            auto const& channel = channels[ch];
            auto const floatData = reinterpret_cast<Tensor const*>(channel.data);

            if(!m_engine->loadToInput(b, offset, floatData, channel.total()))
            {
                return false;
            }

            offset += channel.total();
        }
    }

    return true;
}
}
