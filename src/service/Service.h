#pragma once

#include <rep_SkinCancerDetectorService_source.h>
#include <memory>

#include "common/IEstimated.h"
#include "engines/ITensorEngine.h"
#include "image/IImageConvertor.h"


namespace service
{
class ServiceSettings;
class TensorEngineWorker;
class ImageConvertorWorker;

/**
 * @brief The Service class - receiver of request
 */
class Service : public SkinCancerDetectorServiceSource
{
    Q_OBJECT

public:
    explicit Service(QObject* parent = nullptr);
    ~Service() override;

    /**
     * @brief start all components
     */
    void start();

protected:
    /**
     * @brief request from client
     * @param image - bin data of image
     * @return request info (id - request id, estimateMs - estimated time in ms for handle request, netgative - invalid value)
     */
    SkinCancerDetectorRequestInfo request(QByteArray image) override;

    /**
     * @brief request from client
     * @param imagePath - path to local image
     * @return request info (id - request id, estimateMs - estimated time in ms for handle request, netgative - invalid value)
     */
    SkinCancerDetectorRequestInfo request(QString imagePath) override;

private slots:
    void onSuccess(quint64 id, float positive, float negative);
    void onError(quint64 id, ErrorType type);

private:
    void createComponents();
    void setupService(ServiceSettings const& settings,
                      engines::ITensorEnginePtr const& tensorEngine,
                      image::IImageConvertorPtr const& imageConvertor);
    void enableRemoting(ServiceSettings const& settings);
    void estimate(common::IEstimated* imageConvertor, common::IEstimated* tensorEngine);

    qint64 estimateNextRequest() const;
    quint64 getRequestId();

private:
    TensorEngineWorker* m_tensorEngineWorker = nullptr;
    ImageConvertorWorker* m_imageConvertorWorker = nullptr;

    qint64 m_tensorEngineEstimate = -1;
    qint64 m_imageConvertorEstimate = -1;
    quint64 m_requestId = 0;
};
}
