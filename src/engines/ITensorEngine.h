#pragma once

#include <QtGlobal>

#include <memory>
#include <cstdint>

#include "common/IEstimated.h"
#include "BaseTensorEngineSettings.h"


namespace engines
{
/**
 * @brief The ITensorEngine interface forward data through Neural Network
 */
class ITensorEngine : public common::IEstimated
{
public:
    /**
     * Tensor type
     */
    using Tensor = float;

    virtual ~ITensorEngine() { }

    /**
     * @brief build engine
     * @return success
     */
    virtual bool load(BaseTensorEngineSettings const& settings) = 0;

    /**
     * @brief max batches in TensorEngine(max batches which can forward engine)
     * @return size_t
     */
    virtual size_t maxBatches() const = 0;

    /**
     * @brief input width data
     * @return size_t
     */
    virtual size_t inputWidth() const = 0;

    /**
     * @brief input height data
     * @return size_t
     */
    virtual size_t inputHeight() const = 0;

    /**
     * @brief input channels data
     * @return size_t
     */
    virtual size_t inputChannels() const = 0;

    /**
     * @brief output size after forward
     * @return size_t
     */
    virtual size_t outputSize() const = 0;

    /**
     * @brief count elements in one input batch
     * @return
     */
    virtual size_t batchInputN() const = 0;

    /**
     * @brief count elements in one output batch
     * @return
     */
    virtual size_t batchOutputN() const = 0;

    /**
     * @brief positive index of output data
     * @return
     */
    virtual size_t positiveIndex() const = 0;

    /**
     * @brief negative index of output data
     * @return
     */
    virtual size_t negativeIndex() const = 0;

    /**
     * @brief load to tnput data to device
     * @param batch - number of batch
     * @param offset - index of memory batch where will be record data
     * @param src - source data
     * @param n - count data need to load to device
     * @return bool - success
     */
    virtual bool loadToInput(size_t batch, size_t offset, Tensor const* src, size_t n) = 0;

    /**
     * @brief unload output from device
     * @param batches - count batches to unload
     * @param dst - destination memory
     * @return bool - success
     */
    virtual bool unloadOutput(size_t batches, Tensor* dst) = 0;

    /**
     * @brief forward data
     * @param batches - count batches for forward
     * @return bool - success
     */
    virtual bool infer(size_t batches) = 0;
};

using ITensorEnginePtr = std::shared_ptr<ITensorEngine>;
}

