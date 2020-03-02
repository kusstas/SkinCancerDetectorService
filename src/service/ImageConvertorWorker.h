#pragma once

#include <QObject>
#include <QThreadPool>

#include <atomic>
#include <memory>

#include <rep_SkinCancerDetectorService_source.h>
#include "image/IImageConvertor.h"


namespace image
{
enum class ImageConvertorTypeError;
}

namespace service
{
/**
 * @brief The ImageConvertorWorker class - ImageConvertor worker in thread pool
 */
class ImageConvertorWorker : public QObject
{
    Q_OBJECT

    Q_PROPERTY(bool running READ running WRITE setRunning NOTIFY runningChanged)

    friend class CommonRunnable;

public:
    explicit ImageConvertorWorker(image::IImageConvertorPtr const& imageConvertor,
                                  size_t maxThreads,
                                  QObject* parent = nullptr);
    ~ImageConvertorWorker();

    /**
     * @brief running state (see start/stop)
     * @return
     */
    bool running() const;

    /**
     * @brief queue size
     * @return
     */
    size_t queueSize() const;

    /**
     * @brief max threads
     * @return
     */
    size_t maxThreads() const;

    /**
     * @brief image convertor
     * @return
     */
    image::IImageConvertorPtr const& imageConvertor() const;

public slots:
    /**
     * @brief start wokrer
     * @param maxThreads - max threads in thread pool
     */
    void start();

    /**
     * @brief stop worker, handle all remaining requests
     */
    void stop();

    /**
     * @brief push request
     * @param id - id of requst
     * @param data - image bin data
     */
    void push(quint64 id, QByteArray const& data);

    /**
     * @brief push request
     * @param id - id of requst
     * @param path - path to image
     */
    void push(quint64 id, QString const& path);

signals:
    /**
     * @brief running changed signak
     * @param running
     */
    void runningChanged(bool running);

    /**
     * @brief result signal
     * @param id - id of requst
     * @param data - result
     */
    void result(quint64 id, common::IEngineInputDataPtr const& data);

    /**
     * @brief error signal
     * @param id - id of requst
     * @param type - type of error
     */
    void error(quint64 id, SkinCancerDetectorServiceSource::ErrorType type);

private:
    void setRunning(bool running);

    template <typename Runnuble, typename T>
    void push(quint64 id, T const& data);

    static SkinCancerDetectorServiceSource::ErrorType convert(image::ImageConvertorTypeError type);

private:
    image::IImageConvertorPtr m_imageConvertor = nullptr;
    bool m_running = false;
    bool m_stop = false;
    std::atomic_size_t m_queueSize = 0;

    QThreadPool m_pool{};
};
}
