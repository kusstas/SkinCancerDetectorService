#pragma once

#include <QString>
#include <QByteArray>
#include <memory>

#include "common/IEstimated.h"
#include "common/IEngineInputData.h"
#include "ImageConvertorSettings.h"


namespace image
{
enum class ImageConvertorTypeError
{
    NoError,
    DataIsEmpty,
    FileNotExist,
    ImpossibleDecode,
    MismatchCountChannels,
    TooSmallImageSize,
    System
};

/**
 * @brief The ImageConvertor interface - convert image for pass data to ITensorEngine
 */
class IImageConvertor : public common::IEstimated
{
public:
    virtual ~IImageConvertor() { }

    /**
     * @brief load image convertor from settings
     * @param settings
     * @return
     */
    virtual bool load(ImageConvertorSettings const& settings) = 0;

    /**
     * @brief convert image to data for pass to TensorEngine
     * @param data - binary data
     * @param error - optional out value error
     * @return data loader
     */
    virtual common::IEngineInputDataPtr convert(QByteArray const& data, ImageConvertorTypeError* error = nullptr) const = 0;

    /**
     * @brief convert image to data for pass to TensorEngine
     * @param path - path to image
     * @param error - optional out value error
     * @return data loader
     */
    virtual common::IEngineInputDataPtr convert(QString const& path, ImageConvertorTypeError* error = nullptr) const = 0;
};

using IImageConvertorPtr = std::shared_ptr<IImageConvertor>;
}
