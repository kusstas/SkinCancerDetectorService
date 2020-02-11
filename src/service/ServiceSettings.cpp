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

void ServiceSettings::setUrl(QUrl const& url)
{
    m_url = url;
}
}
