#pragma once

#include "image/IImageConvertor.h"

#include <opencv2/core/mat.hpp>


namespace image
{
namespace opencv
{
/**
 * @brief The ImageConvertor class - convert image for pass data to TensorEngine
 */
class ImageConvertor : public IImageConvertor
{
public:
    ImageConvertor() = default;

public: // IEstimated interface
    qint64 estimate() override;

public: // IImageConvertor interface
    bool load(ImageConvertorSettings const& settings) override;
    common::IEngineInputDataPtr convert(QByteArray const& data, ImageConvertorTypeError *error) const override;
    common::IEngineInputDataPtr convert(QString const& path, ImageConvertorTypeError *error) const override;

private:
    /**
     * @brief prepare opened image to pass to TensorEngine
     * @param source - image
     * @param error - optional out value error
     * @return @return channels
     */
    common::IEngineInputDataPtr prepare(cv::Mat source, ImageConvertorTypeError* error = nullptr) const;

    /**
     * @brief get auto crop size by size ratio in settings
     * @param source - source size
     * @return size
     */
    cv::Size getAutoCropSize(cv::Size const& source) const;

    /**
     * @brief write error to pointer
     * @param dst
     * @param error
     */
    static void writeError(ImageConvertorTypeError* dst, ImageConvertorTypeError error);

private:
    ImageConvertorSettings m_settings{};
};
}
}
