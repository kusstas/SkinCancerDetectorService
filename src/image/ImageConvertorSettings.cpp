#include "ImageConvertorSettings.h"
#include "utils/JsonHelper.h"


namespace image
{
Q_LOGGING_CATEGORY(QLC_IMAGE_SETTINGS, "ImageConvertorSettings")
static utils::JsonHelper const JSON_HELPER(QLC_IMAGE_SETTINGS);

int ImageConvertorSettings::width() const
{
    return m_width;
}

int ImageConvertorSettings::height() const
{
    return m_height;
}

int ImageConvertorSettings::channels() const
{
    return m_channels;
}

float ImageConvertorSettings::zoom() const
{
    return m_zoom;
}

QVector<float> const& ImageConvertorSettings::std() const
{
    return m_std;
}

QVector<float> const& ImageConvertorSettings::mean() const
{
    return m_mean;
}

size_t ImageConvertorSettings::countTestsForEstimate() const
{
    return m_countTestsForEstimate;
}

void ImageConvertorSettings::setWidth(int width)
{
    m_width = width;
}

void ImageConvertorSettings::setHeight(int height)
{
    m_height = height;
}

void ImageConvertorSettings::setChannels(int channels)
{
    m_channels = channels;
}

bool ImageConvertorSettings::parse(QJsonObject const& json)
{
    return JSON_HELPER.getArray(json, "std", m_std, true)
            && JSON_HELPER.getArray(json, "mean", m_mean, true)
            && JSON_HELPER.get(json, "zoom", m_zoom, true)
            && JSON_HELPER.get(json, "countTestsForEstimate", m_countTestsForEstimate, true);
}

bool ImageConvertorSettings::valid() const
{
    return width() > 0
            && height() > 0
            && channels() > 0
            && zoom() >= 1
            && channels() == std().size()
            && channels() == mean().size()
            && countTestsForEstimate() > 0;
}
}
