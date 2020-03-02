#pragma once

class QJsonObject;

namespace common
{
/**
 * @brief The IJsonParsed interface for parse object from json
 */
class IJsonParsed
{
public:
    virtual ~IJsonParsed() { }

    /**
     * @brief parse object from json
     * @param json object
     * @return success
     */
    virtual bool parse(QJsonObject const& json) = 0;
};
}
