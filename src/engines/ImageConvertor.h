#pragma once

#include "ImageConvertorSettings.h"
#include <QByteArray>

#include <opencv2/core/mat.hpp>


namespace engines
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
 * @brief The ImageConvertor class - convert image for pass data to TensorEngine
 */
class ImageConvertor
{
public:
    explicit ImageConvertor(ImageConvertorSettings const& settings);

    /**
     * @brief convert image to data for pass to TensorEngine
     * @param data - binary data
     * @param error - optional out value error
     * @return channels
     */
    QVector<cv::Mat> convert(QByteArray const& data, ImageConvertorTypeError* error = nullptr) const;

    /**
     * @brief convert image to data for pass to TensorEngine
     * @param path - path to image
     * @param error - optional out value error
     * @return channels
     */
    QVector<cv::Mat> convert(QString const& path, ImageConvertorTypeError* error = nullptr) const;

    /**
     * @brief estimate dummy image
     * @return qint64 - nanoseconds
     */
    qint64 estimate() const;

private:
    /**
     * @brief prepare opened image to pass to TensorEngine
     * @param source - image
     * @param error - optional out value error
     * @return @return channels
     */
    QVector<cv::Mat> prepare(cv::Mat source, ImageConvertorTypeError* error = nullptr) const;

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
