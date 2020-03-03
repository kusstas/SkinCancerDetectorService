#pragma once

#include <optional>

#include "common/ISettings.h"
#include "engines/BaseTensorEngineSettings.h"
#include "image/ImageConvertorSettings.h"
#include "service/ServiceSettings.h"


namespace utils
{
struct Settings : public common::ISettings
{
    engines::BaseTensorEngineSettings tensor;
    image::ImageConvertorSettings image;
    service::ServiceSettings service;

public: // IJsonParsed interface
    bool parse(QJsonObject const& json) override;

public: // ISettings interface
    bool valid() const override;
};

class SettingsReader
{
public:
    SettingsReader() = default;

    std::optional<Settings> read() const;
};
}
