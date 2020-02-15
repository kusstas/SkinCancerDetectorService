#include "ServiceSettings.h"


namespace service
{
bool ServiceSettings::valid() const
{
    return url().isValid() && !url().isEmpty();
}

QUrl const& ServiceSettings::url() const
{
    return m_url;
}

size_t ServiceSettings::maxImageConvertorThreads() const
{
    return m_maxImageConvertorThreads;
}

void ServiceSettings::setUrl(QUrl const& url)
{
    m_url = url;
}

void ServiceSettings::setMaxImageConvertorThreads(size_t maxImageConvertorThreads)
{
    m_maxImageConvertorThreads = maxImageConvertorThreads;
}
}
