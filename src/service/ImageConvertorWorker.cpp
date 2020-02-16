#include "ImageConvertorWorker.h"

#include <QLoggingCategory>
#include <QRunnable>

#include "engines/ImageConvertor.h"

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
        auto const result = getResult();

        if (!result.isEmpty())
        {
            emit worker()->result(id(), result);
        }
        else
        {
            emit worker()->error(id());
        }
        worker()->m_queueSize--;
    }

protected:
    virtual QVector<cv::Mat> getResult() = 0;

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
    QVector<cv::Mat> getResult() override
    {
        return worker()->imageConvertor()->convert(m_data);
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
    QVector<cv::Mat> getResult() override
    {
        return worker()->imageConvertor()->convert(m_path);
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
    qCInfo(QLC_IMAGE_WORKER) << "Start requiered, max threads:" << m_pool.maxThreadCount();
    setRunning(true);
}

void ImageConvertorWorker::stop()
{
    qCInfo(QLC_IMAGE_WORKER) << "Stop requiered";

    m_pool.waitForDone();
    setRunning(false);
}

void ImageConvertorWorker::push(quint64 id, QByteArray const& data)
{
    m_queueSize++;
    m_pool.start(new BinImageRunnable(id, data, this));
}

void ImageConvertorWorker::push(quint64 id, QString const& path)
{
    m_queueSize++;
    m_pool.start(new PathImageRunnable(id, path, this));
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
}
