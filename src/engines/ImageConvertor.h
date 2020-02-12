#pragma once

#include "ImageConvertorSettings.h"
#include <QByteArray>

#include <opencv2/core/mat.hpp>


namespace engines
{
/**
 * @brief The ImageConvertor class - convert image for pass data to TensorEngine
 */
class ImageConvertor
{
public:
    explicit ImageConvertor(ImageConvertorSettings const& settings);

    /**
     * @brief convert image to data for pass to TensorEngine
     * @param data - binary data
     * @return channels
     */
    QVector<cv::Mat> convert(QByteArray const& data) const;

    /**
     * @brief convert image to data for pass to TensorEngine
     * @param path - path to image
     * @return channels
     */
    QVector<cv::Mat> convert(QString const& path) const;

    /**
     * @brief estimate dummy image
     * @return qint64 - milliseconds
     */
    qint64 estimate() const;

private:
    /**
     * @brief prepare opened image to pass to TensorEngine
     * @param source - image
     * @return @return channels
     */
    QVector<cv::Mat> prepare(cv::Mat source) const;

    /**
     * @brief get auto crop size by size ratio in settings
     * @param source - source size
     * @return size
     */
    cv::Size getAutoCropSize(cv::Size const& source) const;

private:
    ImageConvertorSettings m_settings{};
};
}
