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
     * @brief max image convertor threads
     * @return
     */
    size_t maxImageConvertorThreads() const;

    /**
     * @brief set url
     * @param url
     */
    void setUrl(QUrl const& url);

    /**
     * @brief set max image convertor threads
     * @warning should be greater than 0
     */
    void setMaxImageConvertorThreads(size_t maxImageConvertorThreads);

private:
    QUrl m_url{};
    size_t m_maxImageConvertorThreads = 0;
};
}
