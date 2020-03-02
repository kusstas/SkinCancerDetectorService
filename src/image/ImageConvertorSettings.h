#pragma once

#include "common/ISettings.h"

#include <QVector>


namespace image
{
/**
 * @brief The ImageConvertorSettings class - settings for ImageConvertor
 */
class ImageConvertorSettings : public common::ISettings
{
public:
    ImageConvertorSettings() = default;

    /**
     * @brief width - necessary width for forward data through TensorEngine
     * @return int - width
     */
    int width() const;

    /**
     * @brief height - necessary height for forward data through TensorEngine
     * @return int - height
     */
    int height() const;

    /**
     * @brief channels - necessary channels for forward data through TensorEngine
     * @return int - channels
     */
    int channels() const;

    /**
     * @brief zoom - zoom of image for crop in center
     * @return
     */
    float zoom() const;

    /**
     * @brief std - scaller for pixel in channels
     * @return array of scaller
     */
    QVector<float> const& std() const;

    /**
     * @brief mean - align for pixel in channels
     * @warning for valid should be equal channels
     * @return array of values
     */
    QVector<float> const& mean() const;

    /**
     * @brief count tests for estimate convert image
     * elapced time will be calculated average
     * @return count
     */
    size_t countTestsForEstimate() const;

    /**
     * @brief set width
     * @warning should be greater than zero
     * @param width
     */
    void setWidth(int width);

    /**
     * @brief set height
     * @warning should be greater than zero
     * @param height
     */
    void setHeight(int height);

    /**
     * @brief set channels
     * @warning should be greater than zero
     * @param channels
     */
    void setChannels(int channels);

public: // IJsonParsed interface
    bool parse(QJsonObject const& json) override;

public: // ISettings interface
    bool valid() const override;

private:
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    float m_zoom = 0;

    QVector<float> m_std{};
    QVector<float> m_mean{};

    size_t m_countTestsForEstimate = 0;
};
}
