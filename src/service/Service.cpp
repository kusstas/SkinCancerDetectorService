#include "Service.h"

#include "utils/ServiceLocator.h"
#include "utils/SettingsReader.h"
#include "TensorEngineWorker.h"
#include "ImageConvertorWorker.h"

#include <QRemoteObjectHost>
#include <QLoggingCategory>


static constexpr auto SETTINGS_PATH = "settings.json";


namespace service
{
Q_LOGGING_CATEGORY(QLC_SERVICE, "Service")

static utils::ServiceLocator serviceLocator;


Service::Service(QObject* parent)
    : SkinCancerDetectorServiceSource(parent)
{
    serviceLocator.init();
    createComponents();
}

Service::~Service()
{
    m_imageConvertorWorker->stop();
    m_tensorEngineWorker->stop();
}

void Service::start()
{
    m_imageConvertorWorker->start();
    m_tensorEngineWorker->start();
}

SkinCancerDetectorRequestInfo Service::request(QByteArray image)
{
    auto const id = getRequestId();
    auto const estimates = estimateNextRequest();

    qCInfo(QLC_SERVICE) << "Request received:" << id << "data size" << image.size() << "estimates" << estimates;

    m_imageConvertorWorker->push(id, image);

    return SkinCancerDetectorRequestInfo{id, estimates};
}

SkinCancerDetectorRequestInfo Service::request(QString imagePath)
{
    auto const id = getRequestId();
    auto const estimates = estimateNextRequest();

    qCInfo(QLC_SERVICE) << "Request received:" << id << "image path" << imagePath << "estimates" << estimates;

    m_imageConvertorWorker->push(id, imagePath);

    return SkinCancerDetectorRequestInfo{id, estimates};
}

void Service::onSuccess(quint64 id, float positive, float negative)
{
    qCInfo(QLC_SERVICE) << "Request handled successfully, id:" << id << "positive:" << positive << "negative:" << negative;

    emit resultReady(id, SkinCancerDetectorResult{positive, negative});
}

void Service::onError(quint64 id, ErrorType type)
{
    qCInfo(QLC_SERVICE) << "Request was failed, id:" << id << "type:" << QMetaEnum::fromType<ErrorType>().key(type);

    emit resultFailed(id, type);
}

void Service::createComponents()
{
    utils::SettingsReader settingsReader;

    auto settings = settingsReader.read();

    if (!settings || settings->valid())
    {
        auto const message = "Reading settings failed";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    serviceLocator.setTensorEngineType(settings->tensor.type());

    // create tensor engine
    auto tensorEngine = serviceLocator.createTensorEngine();
    if (!tensorEngine)
    {
        auto const message = QString("Cannot create tensor engine by type: %1").arg(settings->tensor.type());
        qCCritical(QLC_SERVICE) << qPrintable(message);
        throw std::runtime_error(qPrintable(message));
    }
    else if (!tensorEngine->load(settings->tensor))
    {
        auto const message = "Cannot load tensor engine settings";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    settings->image.setWidth(tensorEngine->inputWidth());
    settings->image.setHeight(tensorEngine->inputHeight());
    settings->image.setChannels(tensorEngine->inputChannels());

    auto imageConvertor = serviceLocator.createImageConvertor();
    if (!imageConvertor)
    {
        auto const message = "Cannot create image convertor";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }
    else if (!imageConvertor->load(settings->image))
    {
        auto const message = "Cannot image convertor settings";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    // setup service
    setupService(settings->service, tensorEngine, imageConvertor);
    estimate(imageConvertor.get(), tensorEngine.get());
}

void Service::setupService(ServiceSettings const& settings,
                           engines::ITensorEnginePtr const& tensorEngine,
                           image::IImageConvertorPtr const& imageConvertor)
{
    if (!settings.valid())
    {
        auto const message = "Service settings is invalid";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    // create image convertor worker
    auto const maxThreads = settings.maxImageConvertorThreads() > 0
            ? settings.maxImageConvertorThreads()
            : std::thread::hardware_concurrency();
    m_imageConvertorWorker = new ImageConvertorWorker(imageConvertor, maxThreads, this);

    // create tensor engine worker
    m_tensorEngineWorker = new TensorEngineWorker(tensorEngine, this);

    connect(m_imageConvertorWorker, &ImageConvertorWorker::result, m_tensorEngineWorker, &TensorEngineWorker::push, Qt::DirectConnection);
    connect(m_imageConvertorWorker, &ImageConvertorWorker::error, this, &Service::onError);
    connect(m_tensorEngineWorker, &TensorEngineWorker::result, this, &Service::onSuccess);
    connect(m_tensorEngineWorker, &TensorEngineWorker::error, this, &Service::onError);

    // enable remoting
    enableRemoting(settings);
}

void Service::enableRemoting(ServiceSettings const& settings)
{
    qCInfo(QLC_SERVICE) << "Trying to enable remoting" << settings.url();

    auto const node = new QRemoteObjectHost(settings.url(), this);

    if(!node->enableRemoting(this))
    {
        auto const message = QString("Enable remoting failed: ")
                + QMetaEnum::fromType<QRemoteObjectHost::ErrorCode>().valueToKey(node->lastError());
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(qPrintable(message));
    }

    qCInfo(QLC_SERVICE) << "Remoting enabled successfully";
}

void Service::estimate(common::IEstimated* imageConvertor, common::IEstimated* tensorEngine)
{
    m_imageConvertorEstimate = imageConvertor->estimate();
    if (m_imageConvertorEstimate < 0)
    {
        auto const message = "Failed to estimate image convertor";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

    m_tensorEngineEstimate = tensorEngine->estimate();
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
    int const batchesTensorProcessing = countTensorProcessing / m_tensorEngineWorker->maxBatches() +
            static_cast<bool>(countTensorProcessing % m_tensorEngineWorker->maxBatches());

    auto const imageTimeProcessing = batchesImageProcessing * m_imageConvertorEstimate;
    auto const tensorTimeProcessing = batchesTensorProcessing * m_tensorEngineEstimate;

    return (imageTimeProcessing + tensorTimeProcessing) / 1000000;
}

quint64 Service::getRequestId()
{
    return ++m_requestId;
}
}
