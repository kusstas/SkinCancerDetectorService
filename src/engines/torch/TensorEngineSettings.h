#pragma once

#include <QString>

#include "engines/BaseTensorEngineSettings.h"


namespace engines
{
namespace torch
{
/**
 * @brief The TensorEngineSettings class contains setting for build TensorEngine
 */
class TensorEngineSettings : public engines::BaseTensorEngineSettings
{
public:
    static void registerSelf(QString const& name);

    /**
     * @brief input width
     * @return
     */
    size_t width() const;

    /**
      * @brief input height
      * @return
      */
    size_t height() const;

    /**
      * @brief input channels
      * @return
      */
    size_t channels() const;

    /**
      * @brief output size
      * @return
      */
    size_t output() const;

    /**
     * @brief model - path to torch model
     * @return
     */
    QString const& modelPath() const;

    /**
     * @brief device for allocate torch
     * @return
     */
    QString const& device() const;

public: // IJsonParsed interface
    bool parse(QJsonObject const& json) override;

public: // ISettings interface
    bool valid() const override;

private:
    size_t m_width = 0;
    size_t m_height = 0;
    size_t m_channels = 0;
    size_t m_output = 0;
    QString m_modelPath{};
    QString m_device{};
};
}
}
