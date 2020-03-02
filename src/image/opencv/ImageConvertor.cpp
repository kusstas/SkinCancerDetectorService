#include "ImageConvertor.h"
#include "engines/ITensorEngine.h"

#include <QLoggingCategory>
#include <QElapsedTimer>
#include <QFile>

#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


namespace image
{
namespace opencv
{
Q_LOGGING_CATEGORY(QLC_OPENCV_CONVERTOR, "OpenCvConvertor")

class EngineInputData : public common::IEngineInputData
{
public:
    EngineInputData(std::vector<cv::Mat>&& data)
        : m_data(std::move(data))
    {
    }

public: // IEngineInputData interface
    bool load(size_t batch, engines::ITensorEngine& dst) override
    {
        size_t offset = 0;
        for (size_t ch = 0; ch < m_data.size(); ++ch)
        {
            auto const& channel = m_data[ch];
            auto const floatData = reinterpret_cast<engines::ITensorEngine::Tensor const*>(channel.data);

            if(!dst.loadToInput(batch, offset, floatData, channel.total()))
            {
                return false;
            }

            offset += channel.total();
        }

        return true;
    }

private:
    std::vector<cv::Mat> m_data{};
};

qint64 ImageConvertor::estimate()
{
    qCInfo(QLC_OPENCV_CONVERTOR) << "Estimate prepare starting";

    auto const estimateSuccess = [] (qint64 milliseconds) {
        qCInfo(QLC_OPENCV_CONVERTOR) << "Estimate prepare completed:" << milliseconds << "nanoseconds";
        return milliseconds;
    };

    auto const estimateFailed = [] () {
        qCInfo(QLC_OPENCV_CONVERTOR) << "Estimate prepare failed";
        return -1;
    };

    cv::Mat source(1000, 1000, CV_8UC3);
    if (source.empty())
    {
        return estimateFailed();
    }

    QElapsedTimer timer;
    timer.start();

    for (size_t i = 0; i < m_settings.countTestsForEstimate(); ++i)
    {
        if(!prepare(source))
        {
            return estimateFailed();
        }
    }

    return estimateSuccess((timer.nsecsElapsed() / m_settings.countTestsForEstimate()) * 2);
}

bool ImageConvertor::load(ImageConvertorSettings const& settings)
{
    if (!settings.valid())
    {
        qCCritical(QLC_OPENCV_CONVERTOR) << "Settings is invalid";
        return false;
    }
    m_settings = settings;

    return true;
}

common::IEngineInputDataPtr ImageConvertor::convert(QByteArray const& data, ImageConvertorTypeError* error) const
{
    if (data.isEmpty())
    {
        writeError(error, ImageConvertorTypeError::DataIsEmpty);
        qCCritical(QLC_OPENCV_CONVERTOR) << "Data is empty";
        return nullptr;
    }

    cv::Mat source = cv::imdecode(cv::_InputArray(data.data(), data.size()), cv::IMREAD_COLOR);

    if (source.empty())
    {
        writeError(error, ImageConvertorTypeError::ImpossibleDecode);
        qCCritical(QLC_OPENCV_CONVERTOR) << "Cannot decode image from binary";
        return nullptr;
    }

    return prepare(source, error);
}

common::IEngineInputDataPtr ImageConvertor::convert(QString const& path, ImageConvertorTypeError* error) const
{
    if (!QFile(path).exists())
    {
        writeError(error, ImageConvertorTypeError::FileNotExist);
        qCCritical(QLC_OPENCV_CONVERTOR) << "File not exists by path:" << path;
        return nullptr;
    }

    cv::Mat source = cv::imread(qPrintable(path));

    if (source.empty())
    {
        writeError(error, ImageConvertorTypeError::ImpossibleDecode);
        qCCritical(QLC_OPENCV_CONVERTOR) << "Cannot open image by path" << path;
        return nullptr;
    }

    return prepare(source, error);
}

common::IEngineInputDataPtr ImageConvertor::prepare(cv::Mat source, ImageConvertorTypeError* error) const
{
    if (source.channels() != m_settings.channels())
    {
        writeError(error, ImageConvertorTypeError::MismatchCountChannels);
        qCCritical(QLC_OPENCV_CONVERTOR) << "Mismacth count channels, setted:" << m_settings.channels()
                                         << "in source:" << source.channels();
        return nullptr;
    }

    if (source.size().width < m_settings.width() || source.size().height < m_settings.height())
    {
        writeError(error, ImageConvertorTypeError::TooSmallImageSize);
        qCCritical(QLC_OPENCV_CONVERTOR) << "Too small image size";
        return nullptr;
    }

    auto crop = getAutoCropSize(source.size());
    crop.height = static_cast<int>(crop.height / m_settings.zoom());
    crop.width = static_cast<int>(crop.width / m_settings.zoom());

    cv::Rect roi;
    roi.x = (source.size().width - crop.width) / 2;
    roi.y = (source.size().height - crop.height) / 2;
    roi.width = crop.width;
    roi.height = crop.height;

    qCDebug(QLC_OPENCV_CONVERTOR) << "Crop image from:" << source.size().width << source.size().height
                                  << "to:" << roi.width << roi.height;
    source = source(roi);

    qCDebug(QLC_OPENCV_CONVERTOR) << "Crop image from:" << source.size().width << source.size().height
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

    return std::make_shared<EngineInputData>(std::move(channels));
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

void ImageConvertor::writeError(ImageConvertorTypeError* dst, ImageConvertorTypeError error)
{
    if (dst)
    {
        *dst = error;
    }
}
}
}
