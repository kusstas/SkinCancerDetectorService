#include "TensorEngineSettings.h"


namespace engines
{
bool TensorEngineBuildSettings::valid() const
{
    return maxBatches() > 0
            && maxWorkspaceSize() > 0
            && !onnxFilePath().isEmpty()
            && !serializedFilePath().isEmpty()
            && countTestsForEstimate() > 0;
}

size_t TensorEngineBuildSettings::maxBatches() const
{
    return m_maxBatches;
}

size_t TensorEngineBuildSettings::maxWorkspaceSize() const
{
    return m_maxWorkspaceSize;
}

QString const& TensorEngineBuildSettings::onnxFilePath() const
{
    return m_onnxFilePath;
}

QString const& TensorEngineBuildSettings::serializedFilePath() const
{
    return m_serializedFilePath;
}

size_t TensorEngineBuildSettings::countTestsForEstimate() const
{
    return m_countTestsForEstimate;
}

void TensorEngineBuildSettings::setMaxBatches(size_t maxBatches)
{
    m_maxBatches = maxBatches;
}

void TensorEngineBuildSettings::setMaxWorkspaceSize(size_t maxWorkspaceSize)
{
    m_maxWorkspaceSize = maxWorkspaceSize;
}

void TensorEngineBuildSettings::setOnnxFilePath(QString const& onnxFilePath)
{
    m_onnxFilePath = onnxFilePath;
}

void TensorEngineBuildSettings::setSerializedFilePath(QString const& serializedFilePath)
{
    m_serializedFilePath = serializedFilePath;
}

void TensorEngineBuildSettings::setCountTestsForEstimate(size_t countTestsForEstimate)
{
    m_countTestsForEstimate = countTestsForEstimate;
}

QDebug operator<<(QDebug d, TensorEngineBuildSettings const& obj)
{
    d << "{"
      << "maxBatches=" << obj.maxBatches()
      << "maxWorkspaceSize=" << obj.maxWorkspaceSize()
      << "onnxFilePath=" << obj.onnxFilePath()
      << "serializedFilePath=" << obj.serializedFilePath()
      << "countTestsForEstimate=" << obj.countTestsForEstimate()
      << "}";

    return d;
}
}
