#pragma once


#include <QtGlobal>

namespace common
{
/**
 * @brief The IEstimated interface for estimate some worker/engine
 */
class IEstimated
{
public:
    virtual ~IEstimated() {}

    /**
     * @brief estimate
     * @return qint64 - nanoseconds of forward (-1 is invalid value)
     */
    virtual qint64 estimate() = 0;
};
}
