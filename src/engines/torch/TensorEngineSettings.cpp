#include "TensorEngineSettings.h"
#include "utils/JsonHelper.h"

#include <QLoggingCategory>


namespace engines
{
namespace torch
{
Q_LOGGING_CATEGORY(QLC_TORCH_SETTINGS, "TorchSettings")
static utils::JsonHelper const JSON_HELPER(QLC_TORCH_SETTINGS);

void TensorEngineSettings::registerSelf(QString const& name)
{
    registerType<TensorEngineSettings>(name);
}

size_t TensorEngineSettings::width() const
{
    return m_width;
}

size_t TensorEngineSettings::height() const
{
    return m_height;
}

size_t TensorEngineSettings::channels() const
{
    return m_channels;
}

size_t TensorEngineSettings::output() const
{
    return m_output;
}

QString const& TensorEngineSettings::modelPath() const
{
    return m_modelPath;
}

QString const& TensorEngineSettings::device() const
{
    return m_device;
}

bool TensorEngineSettings::parse(QJsonObject const& json)
{
    JSON_HELPER.get(json, "device", m_device, false);
    return JSON_HELPER.get(json, "width", m_width, true)
           && JSON_HELPER.get(json, "height", m_height, true)
           && JSON_HELPER.get(json, "channels", m_channels, true)
           && JSON_HELPER.get(json, "output", m_output, true)
           && JSON_HELPER.get(json, "modelPath", m_modelPath, true);
}

bool TensorEngineSettings::valid() const
{
    return BaseTensorEngineSettings::valid()
            && width() > 0
            && height() > 0
            && channels() > 0
            && output() > 0
            && !modelPath().isEmpty();
}
}
}
