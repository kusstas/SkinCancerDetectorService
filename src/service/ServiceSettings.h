#pragma once

#include "common/ISettings.h"

#include <QUrl>


namespace service
{
/**
 * @brief The ServiceSettings class - setting for create service
 */
class ServiceSettings : public common::ISettings
{
public:
    ServiceSettings() = default;

    /**
     * @brief url - url of service for connecting clients
     * @return url
     */
    QUrl const& url() const;

    /**
     * @brief max image convertor threads
     * if less than zero will be equal hardware value
     * @return
     */
    int maxImageConvertorThreads() const;

public: // IJsonParsed interface
    bool parse(const QJsonObject &json) override;

public: // ISettings interface
    bool valid() const override;

private:
    QUrl m_url{};
    int m_maxImageConvertorThreads = 0;
};
}
