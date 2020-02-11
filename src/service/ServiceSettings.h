#pragma once

#include <QUrl>


namespace service
{
/**
 * @brief The ServiceSettings class - setting for create service
 */
class ServiceSettings
{
public:
    ServiceSettings() = default;

    /**
     * @brief valid - settings is valid
     * @return bool value
     */
    bool valid() const;

    /**
     * @brief url - url of service for connecting clients
     * @return url
     */
    QUrl const& url() const;

    /**
     * @brief set url
     * @param url
     */
    void setUrl(QUrl const& url);

private:
    QUrl m_url{};
};
}
