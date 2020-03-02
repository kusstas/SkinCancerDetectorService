#pragma once

#include <QObject>
#include <QList>

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <rep_SkinCancerDetectorService_source.h>

#include "common/IEngineInputData.h"
#include "engines/ITensorEngine.h"


namespace service
{
/**
 * @brief The TensorEngineWorker class - TensorEngine worker in own thread
 */
class TensorEngineWorker : public QObject
{
    Q_OBJECT

    Q_PROPERTY(bool running READ running NOTIFY runningChanged)

public:
    explicit TensorEngineWorker(engines::ITensorEnginePtr const& engine, QObject* parent = nullptr);
    ~TensorEngineWorker();

    /**
     * @brief running - running state
     * @return state
     */
    bool running() const;

    /**
     * @brief queue size
     * @return size
     */
    int queueSize() const;

    /**
     * @brief max batches
     * @return
     */
    size_t maxBatches() const;

public slots:
    /**
     * @brief start worker
     */
    void start();

    /**
     * @brief stop worker
     */
    void stop();

    /**
     * @brief push request to worker
     * @param id - request id
     * @param data
     */
    void push(quint64 id, common::IEngineInputDataPtr const& data);

signals:
    /**
     * @brief emit by changing running state
     * @param running
     */
    void runningChanged(bool running);

    /**
     * @brief result ready signal
     * @param id - request id
     */
    void result(quint64 id, float positive, float negative);

    /**
     * @brief error signal
     * @param id - request id
     */
    void error(quint64 id, SkinCancerDetectorServiceSource::ErrorType type);

private:
    struct Request
    {
        quint64 id;
        common::IEngineInputDataPtr data;
    };

private:
    void setRunning(bool running);

    void run();
    void sendFailed(QList<Request> const& data);
    bool loadData(QList<Request> const& data);

private:
    engines::ITensorEnginePtr m_engine = nullptr;

    std::thread m_thread{};
    std::mutex m_mutex{};
    std::condition_variable m_notifier{};
    bool m_running = false;
    bool m_stop = false;

    QList<Request> m_requests{};
};
}
