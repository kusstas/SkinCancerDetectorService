#include "Service.h"

#include <QRemoteObjectHost>

#include <QLoggingCategory>

#include "SettingsReader.h"
#include "TensorEngineWorker.h"
#include "ImageConvertorWorker.h"

#include "engines/TensorEngine.h"
#include "engines/ImageConvertor.h"


static constexpr auto SETTINGS_PATH = "settings.json";


namespace service
{
Q_LOGGING_CATEGORY(QLC_SERVICE, "Service")

Service::Service(QObject* parent)
    : SkinCancerDetectorServiceSource(parent)
{
    qRegisterMetaType<QVector<cv::Mat>>("QVectorCvMat");

    createComponents();
    estimate();
}

void Service::start()
{
    m_imageConvertorWorker->start();
    m_tensorEngineWorker->start();
}

void Service::request(quint64 id, QByteArray image)
{
    qCInfo(QLC_SERVICE) << "Request received:" << id << "data size" << image.size();

    emit estimatesReady(id, estimateNextRequest());
    m_imageConvertorWorker->push(id, image);
}

void Service::request(quint64 id, QString imagePath)
{
    qCInfo(QLC_SERVICE) << "Request received:" << id << "image path" << imagePath;

    emit estimatesReady(id, estimateNextRequest());
    m_imageConvertorWorker->push(id, imagePath);
}

void Service::createComponents()
{
    SettingsReader settingsReader;

    auto settings = settingsReader.read(SETTINGS_PATH);

    if (!settings.success)
    {
        auto const message = "Reading settings failed";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    // create tensor engine
    m_tensorEngine.reset(new engines::TensorEngine(settings.tensorEngine));

    // create image convertor
    setupImageConvertorSettings(&settings.imageConvertor);
    m_imageConvertor.reset(new engines::ImageConvertor(settings.imageConvertor));

    // setup service
    setupService(&settings.service);
}

void Service::setupImageConvertorSettings(engines::ImageConvertorSettings* settings) const
{
    settings->setHeight(m_tensorEngine->inputHeight());
    settings->setWidth(m_tensorEngine->inputWidth());
    settings->setChannels(m_tensorEngine->inputChannels());
}

void Service::setupService(ServiceSettings const* settings)
{
    if (!settings->valid())
    {
        auto const message = "Service settings is invalid";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    // create image convertor worker
    m_imageConvertorWorker = new ImageConvertorWorker(m_imageConvertor, settings->maxImageConvertorThreads(), this);

    // create tensor engine worker
    m_tensorEngineWorker = new TensorEngineWorker(m_tensorEngine, this);

    connect(m_imageConvertorWorker, &ImageConvertorWorker::result, m_tensorEngineWorker, &TensorEngineWorker::push, Qt::DirectConnection);
    connect(m_imageConvertorWorker, &ImageConvertorWorker::error, this, &Service::resultFailed);
    connect(m_tensorEngineWorker, &TensorEngineWorker::result, this, [this] (quint64 id, float positive, float negative)
    {
        emit resultReady(id, Result{positive, negative});
    });
    connect(m_tensorEngineWorker, &TensorEngineWorker::error, this, &Service::resultFailed);

    // enable remoting
    enableRemoting(settings);
}

void Service::enableRemoting(ServiceSettings const* settings)
{
    qCInfo(QLC_SERVICE) << "Trying to enable remoting" << settings->url();

    auto const node = new QRemoteObjectHost(settings->url(), this);

    if(!node->enableRemoting(this))
    {
        auto const message = QString("Enable remoting failed: ")
                             + QMetaEnum::fromType<QRemoteObjectHost::ErrorCode>().valueToKey(node->lastError());
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(qPrintable(message));
    }

    qCInfo(QLC_SERVICE) << "Remoting enabled successfully";
}

void Service::estimate()
{
    m_imageConvertorEstimate = m_imageConvertor->estimate();
    if (m_imageConvertorEstimate < 0)
    {
        auto const message = "Failed to estimate image convertor";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    m_tensorEngineEstimate = m_tensorEngine->estimateInfer();
    if (m_tensorEngineEstimate < 0)
    {
        auto const message = "Failed to estimate tensor engine";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }
}

qint64 Service::estimateNextRequest() const
{
    int const countImageProcessing = (m_imageConvertorWorker->queueSize() + 1);
    int const countTensorProcessing = countImageProcessing + m_tensorEngineWorker->queueSize();

    int const batchesImageProcessing = countImageProcessing;
    int const batchesTensorProcessing = countTensorProcessing / m_tensorEngine->maxBatches() +
                                       static_cast<bool>(countTensorProcessing % m_tensorEngine->maxBatches());

    auto const imageTimeProcessing = batchesImageProcessing * m_imageConvertorEstimate;
    auto const tensorTimeProcessing = batchesTensorProcessing * m_tensorEngineEstimate;

    return (imageTimeProcessing + tensorTimeProcessing) / 1000000;
}
}
