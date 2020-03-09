#pragma once

#include "engines/BaseTensorEngine.h"
#include "engines/torch/TensorEngineSettings.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/script.h>
#pragma GCC diagnostic pop


namespace engines
{
namespace torch
{
/**
 * @brief The TensorEngine class forward data through Neural Network by Torch Framework
 */
class TensorEngine : public engines::BaseTensorEngine
{
public:
    TensorEngine() = default;

public: // BaseTensorEngine interface
    bool loadImpl(BaseTensorEngineSettings const& settings) override;

public: // ITensorEngine interface
    size_t maxBatches() const override;
    size_t inputWidth() const override;
    size_t inputHeight() const override;
    size_t inputChannels() const override;
    size_t outputSize() const override;
    bool loadToInput(size_t batch, size_t offset, Tensor const*src, size_t n) override;
    bool unloadOutput(size_t batches, Tensor *dst) override;
    bool infer(size_t batches) override;
    size_t batchInputN() const override;
    size_t batchOutputN() const override;

private:
    TensorEngineSettings const* m_settings = nullptr;
    ::torch::NoGradGuard noGrad{};
    c10::optional<c10::Device> m_device = c10::nullopt;
    ::torch::jit::script::Module m_module{};
    std::vector<Tensor> m_input{};
    ::torch::IValue m_output{};
    size_t m_batchInputN = 0;
};
}
}
