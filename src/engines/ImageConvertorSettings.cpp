#include "ImageConvertorSettings.h"


namespace engines
{
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

void ImageConvertorSettings::setZoom(float zoom)
{
    m_zoom = zoom;
}

void ImageConvertorSettings::setStd(QVector<float> const& std)
{
    m_std = std;
}

void ImageConvertorSettings::setMean(QVector<float> const& mean)
{
    m_mean = mean;
}

void ImageConvertorSettings::setCountTestsForEstimate(size_t countTestsForEstimate)
{
    m_countTestsForEstimate = countTestsForEstimate;
}

QDebug operator<<(QDebug d, ImageConvertorSettings const& obj)
{
    d << "{"
      << "width=" << obj.width()
      << "height=" << obj.height()
      << "channels=" << obj.channels()
      << "zoom=" << obj.zoom()
      << "mean=" << obj.mean()
      << "std=" << obj.std()
      << "countTestsForEstimate=" << obj.countTestsForEstimate()
      << "}";

    return d;
}
}
