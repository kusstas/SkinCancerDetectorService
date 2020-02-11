#pragma once

#include <QString>

#include "service/ServiceSettings.h"
#include "engines/ImageConvertorSettings.h"
#include "engines/TensorEngineSettings.h"


class QJsonObject;

namespace service
{
/**
 * @brief The SettingReader class - class for reading settings from json file
 */
class SettingsReader
{
public:
    /**
     * @brief The Settings Result contains all settings for service
     */
    struct Result
    {
        /**
         * @brief success - success of reading
         */
        bool success = false;

        /**
         * @brief service settings
         */
        service::ServiceSettings service{};

        /**
         * @brief tensor engine build settings
         */
        engines::TensorEngineBuildSettings tensorEngine{};

        /**
         * @brief image convertor settings
         */
        engines::ImageConvertorSettings imageConvertor{};
    };

    SettingsReader() = default;

    /**
     * @brief read setting from json file
     * @param path = path to json file
     * @return result
     */
    Result read(QString const& path) const;

private:
    static bool checkRequirements(QJsonObject const* src, QStringList const& list);
    static bool readService(QJsonObject const* src, service::ServiceSettings& dst);
    static bool readTensorEngine(QJsonObject const* src, engines::TensorEngineBuildSettings& dst);
    static bool readImageConvertor(QJsonObject const* src, engines::ImageConvertorSettings& dst);
};
}
