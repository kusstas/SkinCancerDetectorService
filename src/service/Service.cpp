#include "Service.h"

#include <QRemoteObjectHost>

#include <QLoggingCategory>

#include "SettingsReader.h"
#include "TensorEngineWorker.h"

#include "engines/TensorEngine.h"
#include "engines/ImageConvertor.h"


static constexpr auto SETTINGS_PATH = "settings.json";


namespace service
{
Q_LOGGING_CATEGORY(QLC_SERVICE, "Service")

Service::Service(QObject* parent)
    : SkinCancerDetectorServiceSource(parent)
{
    createComponents();
}

void Service::start()
{
    m_tensorEngineWorker->start();
}

void Service::request(quint64 id, QByteArray image)
{
    qCInfo(QLC_SERVICE) << "Request received:" << id << "data size" << image.size();
}

void Service::request(quint64 id, QString imagePath)
{
    qCInfo(QLC_SERVICE) << "Request received:" << id << "image path" << imagePath;
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

    // create tensor engine worker
    m_tensorEngineWorker = new TensorEngineWorker(m_tensorEngine, this);

    // enable remoting
    enableRemoting(&settings.service);
}

void Service::setupImageConvertorSettings(engines::ImageConvertorSettings* settings) const
{
    settings->setHeight(m_tensorEngine->inputHeight());
    settings->setWidth(m_tensorEngine->inputWidth());
    settings->setChannels(m_tensorEngine->inputChannels());
}

void Service::enableRemoting(ServiceSettings const* settings)
{
    qCInfo(QLC_SERVICE) << "Trying to enable remoting";

    if (!settings->valid())
    {
        auto const message = "Url is invalid for enable remoting";
        qCCritical(QLC_SERVICE) << message;
        throw std::runtime_error(message);
    }

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
}
