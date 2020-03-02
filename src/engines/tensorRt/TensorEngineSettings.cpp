#include "TensorEngineSettings.h"
#include "utils/JsonHelper.h"

#include <QLoggingCategory>


namespace engines
{
namespace tensorRt
{
Q_LOGGING_CATEGORY(QLC_TENSOR_RT_SETTINGS, "TensorRtSettings")
static utils::JsonHelper const JSON_HELPER(QLC_TENSOR_RT_SETTINGS);

void TensorEngineSettings::registerSelf(QString const& name)
{
    registerType<TensorEngineSettings>(name);
}

size_t TensorEngineSettings::maxWorkspaceSize() const
{
    return m_maxWorkspaceSize;
}

QString const& TensorEngineSettings::onnxFilePath() const
{
    return m_onnxFilePath;
}

QString const& TensorEngineSettings::serializedFilePath() const
{
    return m_serializedFilePath;
}

bool TensorEngineSettings::parse(QJsonObject const& json)
{
    return JSON_HELPER.get(json, "maxWorkspaceSize", m_maxWorkspaceSize, true)
            && JSON_HELPER.get(json, "onnxFilePath", m_onnxFilePath, true)
            && JSON_HELPER.get(json, "serializedFilePath", m_serializedFilePath, true);
}

bool TensorEngineSettings::valid() const
{
    return engines::BaseTensorEngineSettings::valid()
            && maxWorkspaceSize() > 0
            && !onnxFilePath().isEmpty()
            && !serializedFilePath().isEmpty();
}
}
}
