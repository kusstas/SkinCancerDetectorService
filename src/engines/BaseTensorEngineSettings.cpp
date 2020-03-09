#include "BaseTensorEngineSettings.h"
#include "utils/JsonHelper.h"

#include <QCoreApplication>
#include <QRegularExpression>
#include <QLoggingCategory>
#include <QJsonObject>
#include <QJsonValue>
#include <QMetaEnum>

#define ARG_TYPE_KEY   "tensor-engine"
#define ARG_TYPE_VALUE "value"


namespace engines
{
Q_LOGGING_CATEGORY(QLC_BASE_TENSOR_SETTINGS, "BaseTensorSettings")

static utils::JsonHelper const JSON_HELPER(QLC_BASE_TENSOR_SETTINGS);
static constexpr auto ARG_TYPE_REG = "--" ARG_TYPE_KEY "=(?<" ARG_TYPE_VALUE ">\\w+)";


BaseTensorEngineSettings::TypesConstructors BaseTensorEngineSettings::m_typesConstructors{};

class CommonTensorEngineSettings : public BaseTensorEngineSettings
{   
public:
    CommonTensorEngineSettings()
        : BaseTensorEngineSettings(false)
    {
        auto args = QCoreApplication::arguments();

        QRegularExpression reg(ARG_TYPE_REG);
        for (int i = 1; i < args.size(); ++i)
        {
            auto match = reg.match(args[i]);
            if (match.hasMatch())
            {
                m_type = match.captured(ARG_TYPE_VALUE);
                break;
            }
        }
    }

    bool parse(QJsonObject const& json) override
    {
        if (m_type.isEmpty())
        {
            if(!JSON_HELPER.get(json, "type", m_type, true))
            {
                return false;
            }
        }
        return JSON_HELPER.get(json, "maxBatches", m_maxBatches, true)
                && JSON_HELPER.get(json, "positiveIndex", m_positiveIndex, true)
                && JSON_HELPER.get(json, "negativeIndex", m_negativeIndex, true)
                && JSON_HELPER.get(json, "countTestsForEstimate", m_countTestsForEstimate, true);
    }

    // TensorEngineSettings interface
public:
    bool valid() const override
    {
        return m_typesConstructors.contains(type())
                && maxBatches() > 0
                && countTestsForEstimate() > 0
                && positiveIndex() >= 0
                && negativeIndex() >= 0
                && positiveIndex() != negativeIndex();
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

    size_t positiveIndex() const override
    {
        return m_positiveIndex;
    }

    size_t negativeIndex() const override
    {
        return m_negativeIndex;
    }

private:
    QString m_type{};
    size_t m_maxBatches = 0;
    size_t m_countTestsForEstimate = 0;
    size_t m_positiveIndex = 0;
    size_t m_negativeIndex = 0;
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

size_t BaseTensorEngineSettings::positiveIndex() const
{
    return m_instance->positiveIndex();
}

size_t BaseTensorEngineSettings::negativeIndex() const
{
    return m_instance->negativeIndex();
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
    return m_instance ? m_instance->valid() : false;
}
}
