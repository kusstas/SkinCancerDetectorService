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
     * if less than zero will be equal hardware value
     * @return
     */
    int maxImageConvertorThreads() const;

    /**
     * @brief set url
     * @param url
     */
    void setUrl(QUrl const& url);

    /**
     * @brief set max image convertor threads
     */
    void setMaxImageConvertorThreads(int maxImageConvertorThreads);

private:
    QUrl m_url{};
    int m_maxImageConvertorThreads = 0;
};
}
