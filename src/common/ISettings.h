#pragma once

#include "IJsonParsed.h"


namespace common
{
/**
 * @brief The ISettings interface for hold and parse settings
 */
class ISettings : public IJsonParsed
{
public:
    /**
     * @brief valid object - setting can be passed to init
     * @return bool
     */
    virtual bool valid() const = 0;
};
}
