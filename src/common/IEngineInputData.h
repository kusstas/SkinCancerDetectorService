#pragma once

#include <cstddef>
#include <memory>


namespace engines
{
class ITensorEngine;
}

namespace common
{
/**
 * @brief The IEngineInputData interface for load data to engine
 */
class IEngineInputData
{
public:
    virtual ~IEngineInputData() { }

    /**
     * @brief load data to engine
     * @param batch - number of batch
     * @param dst - destination
     * @return success
     */
    virtual bool load(size_t batch, engines::ITensorEngine& dst) = 0;
};

using IEngineInputDataPtr = std::shared_ptr<common::IEngineInputData>;
}
