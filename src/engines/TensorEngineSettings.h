#pragma once

#include <cstdint>
#include <QString>
#include <QDebug>


namespace engines
{
/**
 * @brief The TensorEngineSettings class contains setting for build TensorEngine
 */
class TensorEngineBuildSettings
{
public:
    TensorEngineBuildSettings() = default;

    /**
     * @brief valid object - setting can be passed to TensorEngine
     * @return bool
     */
    bool valid() const;

    /**
     * @brief max batches in TensorEngine(max batches which can forward engine)
     * has value only when build engine
     * @return bool
     */
    size_t maxBatches() const;

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

    /**
     * @brief count tests for estimate infer
     * elapced time will be calculated average
     * @return count
     */
    size_t countTestsForEstimate() const;

    /**
     * @brief set max batches
     * @param maxBatches should be greater than zero
     */
    void setMaxBatches(size_t maxBatches);

    /**
     * @brief set max workspace size
     * @param maxWorkspaceSize should be greater than zero
     */
    void setMaxWorkspaceSize(size_t maxWorkspaceSize);

    /**
     * @brief set onnx file path
     * @param onnxFilePath - should be not empty
     */
    void setOnnxFilePath(QString const& onnxFilePath);

    /**
     * @brief set serialized file path
     * @param serializedFilePath - should be not empty
     */
    void setSerializedFilePath(QString const& serializedFilePath);

    /**
     * @brief set count tests for estimate infer
     * @param countTestsForEstimate should be greater than 0
     */
    void setCountTestsForEstimate(size_t countTestsForEstimate);

private:
    size_t m_maxBatches = 0;
    size_t m_maxWorkspaceSize = 0;

    QString m_onnxFilePath{};
    QString m_serializedFilePath{};

    size_t m_countTestsForEstimate = 0;
};

QDebug operator<<(QDebug d, TensorEngineBuildSettings const& obj);
}
