#pragma once

#include "engines/ITensorEngine.h"
#include "engines/BaseTensorEngineSettings.h"


namespace engines
{
/**
 * @brief The TensorEngine class forward data through Neural Network by Torch Framework
 */
class BaseTensorEngine : public engines::ITensorEngine
{
public:
    BaseTensorEngine() = default;

    BaseTensorEngineSettings const& settings() const;

public: // IEstimated interface
    qint64 estimate() override;

public: // ITensorEngine interface
    bool load(BaseTensorEngineSettings const& settings) override;
    size_t positiveIndex() const override;
    size_t negativeIndex() const override;

protected:
    virtual bool loadImpl(BaseTensorEngineSettings const& settings);

    bool validateLoadInput(size_t batch, size_t offset, Tensor const* src, size_t n) const;
    bool validateLoadOutput(size_t batches, Tensor* dst) const;
    bool validateInfer(size_t batches) const;

private:
    BaseTensorEngineSettings m_settings{};
};
}
