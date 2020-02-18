#include "ImageConvertorWorker.h"

#include "engines/ImageConvertor.h"

#include <QLoggingCategory>
#include <QRunnable>

namespace service
{
Q_LOGGING_CATEGORY(QLC_IMAGE_WORKER, "ImageConvertorWorker")

class CommonRunnable : public QRunnable
{
public:
    CommonRunnable(quint64 id, ImageConvertorWorker* worker)
        : m_id(id)
        , m_worker(worker)
    {
        setAutoDelete(true);
    }

    qint64 id() const
    {
        return m_id;
    }

    ImageConvertorWorker* worker() const
    {
        return m_worker;
    }

    void run() override
    {
        engines::ImageConvertorTypeError error = engines::ImageConvertorTypeError::NoError;
        auto const result = getResult(&error);

        if (!result.isEmpty())
        {
            emit worker()->result(id(), result);
        }
        else
        {
            emit worker()->error(id(), ImageConvertorWorker::convert(error));
        }
        worker()->m_queueSize--;
    }

protected:
    virtual QVector<cv::Mat> getResult(engines::ImageConvertorTypeError* error) = 0;

private:
    quint64 m_id = 0;
    ImageConvertorWorker* m_worker = nullptr;
};

class BinImageRunnable : public CommonRunnable
{
public:
    BinImageRunnable(quint64 id, QByteArray const& data, ImageConvertorWorker* worker)
        : CommonRunnable(id, worker)
        , m_data(data)
    {
    }

protected:
    QVector<cv::Mat> getResult(engines::ImageConvertorTypeError* error) override
    {
        return worker()->imageConvertor()->convert(m_data, error);
    }

private:
    QByteArray m_data{};
};

class PathImageRunnable : public CommonRunnable
{
public:
    PathImageRunnable(quint64 id, QString const& path, ImageConvertorWorker* worker)
        : CommonRunnable(id, worker)
        , m_path(path)
    {
    }

protected:
    QVector<cv::Mat> getResult(engines::ImageConvertorTypeError* error) override
    {
        return worker()->imageConvertor()->convert(m_path, error);
    }

private:
    QString m_path{};
};

ImageConvertorWorker::ImageConvertorWorker(std::shared_ptr<engines::ImageConvertor> const& imageConvertor, size_t maxThreads, QObject* parent)
    : QObject(parent)
    , m_imageConvertor(imageConvertor)
{
    m_pool.setMaxThreadCount(maxThreads);
}

ImageConvertorWorker::~ImageConvertorWorker()
{
    stop();
}

bool ImageConvertorWorker::running() const
{
    return m_running;
}

size_t ImageConvertorWorker::queueSize() const
{
    return m_queueSize;
}

size_t ImageConvertorWorker::maxThreads() const
{
    return m_pool.maxThreadCount();
}

engines::ImageConvertor* ImageConvertorWorker::imageConvertor() const
{
    return m_imageConvertor.get();
}

void ImageConvertorWorker::start()
{
    if (running())
    {
        qCWarning(QLC_IMAGE_WORKER) << "Already started";
        return;
    }
    if (m_stop)
    {
        qCWarning(QLC_IMAGE_WORKER) << "Stop in proggress";
        return;
    }

    qCInfo(QLC_IMAGE_WORKER) << "Start requiered, max threads:" << m_pool.maxThreadCount();

    m_stop = false;
    setRunning(true);
}

void ImageConvertorWorker::stop()
{
    if (!running())
    {
        qCWarning(QLC_IMAGE_WORKER) << "Not started";
        return;
    }
    if (m_stop)
    {
        qCWarning(QLC_IMAGE_WORKER) << "Stop already requiered";
        return;
    }

    qCInfo(QLC_IMAGE_WORKER) << "Stop requiered";

    m_stop = true;
    m_pool.waitForDone();
    setRunning(false);
}

template <typename Runnuble, typename T>
void ImageConvertorWorker::push(quint64 id, T const& data)
{
    if (m_stop)
    {
        qCWarning(QLC_IMAGE_WORKER) << "Reject request by stop" << id;
        error(id, SkinCancerDetectorServiceSource::StopService);
    }
    else
    {
        m_queueSize++;
        m_pool.start(new Runnuble(id, data, this));
    }
}

void ImageConvertorWorker::push(quint64 id, QByteArray const& data)
{
    push<BinImageRunnable>(id, data);
}

void ImageConvertorWorker::push(quint64 id, QString const& path)
{
    push<PathImageRunnable>(id, path);
}

void ImageConvertorWorker::setRunning(bool running)
{
    if (m_running == running)
    {
        return;
    }

    qCInfo(QLC_IMAGE_WORKER) << "Set running state:" << running;

    m_running = running;
    emit runningChanged(m_running);
}

SkinCancerDetectorServiceSource::ErrorType ImageConvertorWorker::convert(engines::ImageConvertorTypeError type)
{
    using ICTE = engines::ImageConvertorTypeError;

    switch (type) {
    case ICTE::NoError:
        return SkinCancerDetectorServiceSource::NoError;
    case ICTE::DataIsEmpty:
        return SkinCancerDetectorServiceSource::DataIsEmpty;
    case ICTE::FileNotExist:
        return SkinCancerDetectorServiceSource::FileNotExist;
    case ICTE::ImpossibleDecode:
        return SkinCancerDetectorServiceSource::ImpossibleDecode;
    case ICTE::MismatchCountChannels:
        return SkinCancerDetectorServiceSource::MismatchCountChannels;
    case ICTE::TooSmallImageSize:
        return SkinCancerDetectorServiceSource::TooSmallImageSize;
    case ICTE::System:
        return SkinCancerDetectorServiceSource::System;
    default:
        break;
    }

    return SkinCancerDetectorServiceSource::System;
}
}
