#include "ImageConvertor.h"

#include <QLoggingCategory>
#include <QElapsedTimer>

#include <exception>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace engines
{
Q_LOGGING_CATEGORY(QLC_IMAGE_CONVERTOR, "ImageConvertor")

ImageConvertor::ImageConvertor(ImageConvertorSettings const& settings)
{
    if (!settings.valid())
    {
        auto const msg = "Invalid setting for create image convertor";
        qCCritical(QLC_IMAGE_CONVERTOR) << msg << "settings:" << settings;
        throw std::runtime_error(msg);
    }

    m_settings = settings;

    qCInfo(QLC_IMAGE_CONVERTOR) << "Settings -" << settings;
}

QVector<cv::Mat> ImageConvertor::convert(QByteArray const& data) const
{
    cv::Mat source = cv::imdecode(cv::_InputArray(data.data(), data.size()), cv::IMREAD_COLOR);

    if (source.empty())
    {
        qCCritical(QLC_IMAGE_CONVERTOR) << "Cannot decode image from binary";
        return {};
    }

    return prepare(source);
}

QVector<cv::Mat> ImageConvertor::convert(QString const& path) const
{
    cv::Mat source = cv::imread(qPrintable(path));

    if (source.empty())
    {
        qCCritical(QLC_IMAGE_CONVERTOR) << "Cannot open image by path" << path;
        return {};
    }

    return prepare(source);
}

qint64 ImageConvertor::estimate() const
{
    qCInfo(QLC_IMAGE_CONVERTOR) << "Estimate prepare starting";

    auto const estimateSuccess = [] (qint64 milliseconds) {
        qCInfo(QLC_IMAGE_CONVERTOR) << "Estimate prepare completed:" << milliseconds << "nanoseconds";
        return milliseconds;
    };

    auto const estimateFailed = [] () {
        qCInfo(QLC_IMAGE_CONVERTOR) << "Estimate prepare failed";
        return -1;
    };


    cv::Mat source(1920, 1080, CV_8UC3);
    if (source.empty())
    {
        return estimateFailed();
    }

    QElapsedTimer timer;
    timer.start();

    for (size_t i = 0; i < m_settings.countTestsForEstimate(); ++i)
    {
        if(prepare(source).isEmpty())
        {
            return estimateFailed();
        }
    }

    return estimateSuccess(timer.nsecsElapsed() / m_settings.countTestsForEstimate());
}

QVector<cv::Mat> ImageConvertor::prepare(cv::Mat source) const
{
    if (source.channels() != m_settings.channels())
    {
        qCCritical(QLC_IMAGE_CONVERTOR) << "Mismacth count channels, setted:" << m_settings.channels()
                                        << "in source:" << source.channels();
        return {};
    }

    auto crop = getAutoCropSize(source.size());
    crop.height = static_cast<int>(crop.height / m_settings.zoom());
    crop.width = static_cast<int>(crop.width / m_settings.zoom());

    cv::Rect roi;
    roi.x = (source.size().width - crop.width) / 2;
    roi.y = (source.size().height - crop.height) / 2;
    roi.width = crop.width;
    roi.height = crop.height;

    qCDebug(QLC_IMAGE_CONVERTOR) << "Crop image from:" << source.size().width << source.size().height
                                 << "to:" << roi.width << roi.height;
    source = source(roi);

    qCDebug(QLC_IMAGE_CONVERTOR) << "Crop image from:" << source.size().width << source.size().height
                                 << "to:" << m_settings.width() << m_settings.height();
    cv::resize(source, source, cv::Size(m_settings.width(), m_settings.height()));

    std::vector<cv::Mat> channels;
    cv::split(source, channels);

    for (size_t ch = 0; ch < channels.size(); ++ch)
    {
        channels[ch].convertTo(channels[ch], CV_32FC1);
        channels[ch] /= 255;
        channels[ch] -= m_settings.mean()[ch];
        channels[ch] /= m_settings.std()[ch];
    }

    return {channels.begin(), channels.end()};
}

cv::Size ImageConvertor::getAutoCropSize(cv::Size const& source) const
{
    cv::Size crop(0, 0);

    if (m_settings.height() > m_settings.width())
    {
        auto const r = static_cast<float>(m_settings.width()) / m_settings.height();
        auto const a = r * source.height;

        if (a <= source.width)
        {
            crop.height = source.height;
            crop.width = static_cast<int>(a);
        }
        else
        {
            crop.width = source.width;
            crop.height = static_cast<int>(crop.width / r);
        }
    }
    else
    {
        auto const r = static_cast<float>(m_settings.height()) / m_settings.width();
        auto const a = r * source.width;

        if (a <= source.height)
        {
            crop.width = source.width;
            crop.height = static_cast<int>(a);
        }
        else
        {
            crop.height = source.height;
            crop.width = static_cast<int>(crop.height / r);
        }
    }

    return crop;
}
}
