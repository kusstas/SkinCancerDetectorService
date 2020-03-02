#include "SettingsReader.h"
#include "JsonHelper.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonParseError>

#include <QLoggingCategory>

namespace utils
{
Q_LOGGING_CATEGORY(QLC_SETTINGS, "Settings")
static utils::JsonHelper const JSON_HELPER(QLC_SETTINGS);

static constexpr auto FILE_NAME = "settings.json";

bool Settings::parse(QJsonObject const& json)
{
    return JSON_HELPER.get(json, "nn", &tensor, true)
            && JSON_HELPER.get(json, "image", &image, true)
            && JSON_HELPER.get(json, "service", &service, true);
}

bool Settings::valid() const
{
    return tensor.valid() && image.valid() && service.valid();
}

std::optional<Settings> SettingsReader::read() const
{
    qCInfo(QLC_SETTINGS) << "Trying open settings:" << FILE_NAME;

    QFile file(FILE_NAME);

    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        qCCritical(QLC_SETTINGS) << "Cannot open file:" << FILE_NAME << "error:" << file.errorString();
        return std::nullopt;
    }

    QJsonParseError jError;
    auto root = QJsonDocument::fromJson(file.readAll(), &jError).object();

    if (jError.error != QJsonParseError::NoError)
    {
        qCCritical(QLC_SETTINGS) << "Cannot parse file:" << FILE_NAME << "error:" << jError.errorString();
        return std::nullopt;
    }

    Settings settings;
    if (!JSON_HELPER.get(root, &settings, true))
    {
        return std::nullopt;
    }

    qCInfo(QLC_SETTINGS) << "Open settings successfully";

    return settings;
}
}
