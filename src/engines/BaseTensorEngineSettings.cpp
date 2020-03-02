#include "BaseTensorEngineSettings.h"
#include "utils/JsonHelper.h"

#include <QLoggingCategory>
#include <QJsonObject>
#include <QJsonValue>
#include <QMetaEnum>


namespace engines
{
Q_LOGGING_CATEGORY(QLC_BASE_TENSOR_SETTINGS, "BaseTensorSettings")

static utils::JsonHelper const JSON_HELPER(QLC_BASE_TENSOR_SETTINGS);
BaseTensorEngineSettings::TypesConstructors BaseTensorEngineSettings::m_typesConstructors{};

class CommonTensorEngineSettings : public BaseTensorEngineSettings
{
    // ISettings interface
public:
    CommonTensorEngineSettings()
        : BaseTensorEngineSettings(false)
    {
    }

    bool parse(QJsonObject const& json) override
    {
        return JSON_HELPER.get(json, "type", m_type, true)
                && JSON_HELPER.get(json, "maxBatches", m_maxBatches, true)
                && JSON_HELPER.get(json, "countTestsForEstimate", m_countTestsForEstimate, true);
    }

    // TensorEngineSettings interface
public:
    bool valid() const override
    {
        return false;
    }

    QString const& type() const override
    {
        return m_type;
    }

    size_t maxBatches() const override
    {
        return m_maxBatches;
    }

    size_t countTestsForEstimate() const override
    {
        return m_countTestsForEstimate;
    }

private:
    QString m_type{};
    size_t m_maxBatches = 0;
    size_t m_countTestsForEstimate = 0;
};

void BaseTensorEngineSettings::registerType(QString const& type, TypeConstructor const& constructor)
{
    m_typesConstructors.insert(type, constructor);
}

BaseTensorEngineSettings::BaseTensorEngineSettings()
    : BaseTensorEngineSettings(true)
{
}

BaseTensorEngineSettings::BaseTensorEngineSettings(bool makeCommon)
{
    if (makeCommon)
    {
        m_instance = std::make_unique<CommonTensorEngineSettings>();
    }
}

QString const& BaseTensorEngineSettings::type() const
{
    return m_instance->type();
}

size_t BaseTensorEngineSettings::maxBatches() const
{
    return m_instance->maxBatches();
}

size_t BaseTensorEngineSettings::countTestsForEstimate() const
{
    return m_instance->countTestsForEstimate();
}

bool BaseTensorEngineSettings::parse(QJsonObject const& json)
{
    if (m_instance.get() == nullptr)
    {
        return true;
    }

    if (JSON_HELPER.get(json, m_instance.get(), true))
    {
        auto const constructor = m_typesConstructors.value(type(), {});

        if (constructor)
        {
            auto newInstance = constructor();

            if (newInstance)
            {
                newInstance->m_instance = m_instance;
                if (JSON_HELPER.get(json, type(), newInstance.get(), true))
                {
                    m_instance = newInstance;
                    return true;
                }
            }
        }
        else
        {
            qCCritical(QLC_BASE_TENSOR_SETTINGS) << "Not registered type" << type();
        }
    }

    return false;
}

bool BaseTensorEngineSettings::valid() const
{
    return m_typesConstructors.contains(type()) && maxBatches() > 0 && countTestsForEstimate() > 0;
}
}
