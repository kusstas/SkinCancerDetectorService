#pragma once

#include <cstdint>
#include <QString>

#include "engines/BaseTensorEngineSettings.h"


namespace engines
{
namespace tensorRt
{
/**
 * @brief The TensorEngineSettings class contains setting for build TensorEngine
 */
class TensorEngineSettings : public engines::BaseTensorEngineSettings
{
public:
    static void registerSelf(QString const& name);

    /**
     * @brief max workspace size for TensorEngine
     * has value only when build engine
     * @return size_t
     */
    size_t maxWorkspaceSize() const;

    /**
     * @brief onnx file fath for build engine
     * has value only when build engine
     * @return QString
     */
    QString const& onnxFilePath() const;

    /**
     * @brief serialized file path for TensorEngine
     * @return QString
     */
    QString const& serializedFilePath() const;


public:  // IJsonParsed interface
    bool parse(QJsonObject const& json) override;

public: // ISettings interface
    bool valid() const override;

private:
    size_t m_maxWorkspaceSize = 0;
    QString m_onnxFilePath{};
    QString m_serializedFilePath{};
};
}
}
