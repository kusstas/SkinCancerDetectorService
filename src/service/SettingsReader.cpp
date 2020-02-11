#include "SettingsReader.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QJsonArray>
#include <QJsonParseError>

#include <QLoggingCategory>


namespace service
{
Q_LOGGING_CATEGORY(QLC_SETTINGS, "SettingsReader")

SettingsReader::Result SettingsReader::read(QString const& path) const
{
    qCInfo(QLC_SETTINGS) << "Start reading settings from:" << path;

    Result result;
    QFile file(path);

    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        qCCritical(QLC_SETTINGS) << "Cannot open settings file:" << path << "error:" << file.errorString();
        return result;
    }

    QJsonParseError error;
    QJsonObject root = QJsonDocument::fromJson(file.readAll(), &error).object();

    if (error.error != QJsonParseError::NoError)
    {
        qCCritical(QLC_SETTINGS) << "Error parse:" << error.errorString();
        return result;
    }

    if (root.empty())
    {
        qCCritical(QLC_SETTINGS) << "Nothing constains in root";
        return result;
    }


    // read service settings
    {
        auto object = root["service"].toObject();
        if (!readService(&object, result.service))
        {
            qCCritical(QLC_SETTINGS) << "Parse \"service\" failed";
            return result;
        }
    }

    // read tensor engine settings
    {
        auto object = root["tensor"].toObject();
        if (!readTensorEngine(&object, result.tensorEngine))
        {
            qCCritical(QLC_SETTINGS) << "Parse \"tensor\" failed";
            return result;
        }
    }

    // read image convertor settings
    {
        auto object = root["image"].toObject();
        if (!readImageConvertor(&object, result.imageConvertor))
        {
            qCCritical(QLC_SETTINGS) << "Parse \"image\" failed";
            return result;
        }
    }

    qCInfo(QLC_SETTINGS) << "Reading settings from:" << path << "completed";

    result.success = true;
    return result;
}

bool SettingsReader::checkRequirements(QJsonObject const* src, QStringList const& list)
{
    QStringList lacks;

    for (auto const& key : list)
    {
        if (!src->contains(key))
        {
            lacks.append(key);
        }
    }

    if (!lacks.empty())
    {
        qCCritical(QLC_SETTINGS) << "Lacks this keys:" << lacks;
    }

    return lacks.empty();
}

bool SettingsReader::readService(QJsonObject const* src, service::ServiceSettings& dst)
{
    if (src->empty())
    {
        return false;
    }

    auto const URL_KEY = "url";

    if (!checkRequirements(src, {URL_KEY}))
    {
        return false;
    }

    dst.setUrl(QUrl((*src)[URL_KEY].toString()));

    return true;
}

bool SettingsReader::readTensorEngine(QJsonObject const* src, engines::TensorEngineBuildSettings& dst)
{
    if (src->empty())
    {
        return false;
    }

    auto const MX_WORKSPACE_KEY = "maxWorkspaceSize";
    auto const MX_BATCHES_KEY = "maxBatches";
    auto const ONNX_KEY = "onnxFilePath";
    auto const SERIALIZED_KEY = "serializedFilePath";

    if (!checkRequirements(src, {MX_WORKSPACE_KEY, MX_BATCHES_KEY, ONNX_KEY, SERIALIZED_KEY}))
    {
        return false;
    }

    dst.setMaxWorkspaceSize((*src)[MX_WORKSPACE_KEY].toInt());
    dst.setMaxBatches((*src)[MX_BATCHES_KEY].toInt());
    dst.setOnnxFilePath((*src)[ONNX_KEY].toString());
    dst.setSerializedFilePath((*src)[SERIALIZED_KEY].toString());

    return true;
}

bool SettingsReader::readImageConvertor(QJsonObject const* src, engines::ImageConvertorSettings& dst)
{
    if (src->empty())
    {
        return false;
    }

    auto const STD_KEY = "std";
    auto const MEAN_KEY = "mean";
    auto const ZOOM_KEY = "zoom";

    if (!checkRequirements(src, {STD_KEY, MEAN_KEY, ZOOM_KEY}))
    {
        return false;
    }

    QVector<float> mean, std;

    for (auto const& value : (*src)[STD_KEY].toArray())
    {
        std.append(static_cast<float>(value.toDouble()));
    }
    for (auto const& value : (*src)[MEAN_KEY].toArray())
    {
        mean.append(static_cast<float>(value.toDouble()));
    }

    dst.setStd(std);
    dst.setMean(mean);
    dst.setZoom(static_cast<float>((*src)[ZOOM_KEY].toDouble()));

    return true;
}
}
