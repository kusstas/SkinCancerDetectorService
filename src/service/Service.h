#pragma once

#include <rep_SkinCancerDetectorService_source.h>

#include <memory>


namespace engines
{
class TensorEngine;
class ImageConvertor;
class ImageConvertorSettings;
}

namespace service
{
class ServiceSettings;
class TensorEngineWorker;

class Service : protected SkinCancerDetectorServiceSource
{
    Q_OBJECT

public:
    explicit Service(QObject* parent = nullptr);

    void start();

protected:
    void request(quint64 id, QByteArray image) override;
    void request(quint64 id, QString imagePath) override;

private:
    void createComponents();
    void setupImageConvertorSettings(engines::ImageConvertorSettings* settings) const;
    void enableRemoting(ServiceSettings const* settings);

private:
    std::shared_ptr<engines::TensorEngine> m_tensorEngine = nullptr;
    std::shared_ptr<engines::ImageConvertor> m_imageConvertor = nullptr;

    TensorEngineWorker* m_tensorEngineWorker = nullptr;
};
}
