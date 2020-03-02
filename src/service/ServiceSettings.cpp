#include "ServiceSettings.h"
#include "utils/JsonHelper.h"

#include <QLoggingCategory>


namespace service
{
Q_LOGGING_CATEGORY(QLC_SERVICE_SETTINGS, "ServiceSettings")
static utils::JsonHelper const JSON_HELPER(QLC_SERVICE_SETTINGS);

QUrl const& ServiceSettings::url() const
{
    return m_url;
}

int ServiceSettings::maxImageConvertorThreads() const
{
    return m_maxImageConvertorThreads;
}

bool ServiceSettings::parse(QJsonObject const& json)
{
    QString url;
    return JSON_HELPER.get(json, "url", url, true)
            && JSON_HELPER.get(json, "maxImageConvertorThreads", m_maxImageConvertorThreads, true)
            && (m_url = QUrl(url), true);
}

bool ServiceSettings::valid() const
{
    return url().isValid() && !url().isEmpty();
}
}
