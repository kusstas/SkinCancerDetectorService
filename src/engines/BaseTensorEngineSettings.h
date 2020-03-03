#pragma once

#include <functional>
#include <type_traits>
#include <memory>

#include <QMap>

#include "common/ISettings.h"


namespace engines
{
/**
 * @brief The BaseTensorEngineSettings class - common settings for tensor engines
 */
class BaseTensorEngineSettings : public common::ISettings
{
    friend class CommonTensorEngineSettings;

public:
    using TypeConstructor = std::function<std::shared_ptr<BaseTensorEngineSettings>()>;
    static void registerType(QString const& type, TypeConstructor const& constructor);

    template <typename T, typename... Args>
    static void registerType(QString const& type, Args const&... args)
    {
        return registerType(type, [args...] { return std::make_unique<T>(args...); });
    }

public:
    BaseTensorEngineSettings();

    /**
     * @brief type of settings for engine
     * @return
     */
    virtual QString const& type() const;

    /**
     * @brief max batches in TensorEngine(max batches which can forward engine)
     * has value only when build engine
     * @return bool
     */
    virtual size_t maxBatches() const;

    /**
     * @brief count tests for estimate infer
     * elapced time will be calculated average
     * @return count
     */
    virtual size_t countTestsForEstimate() const;

    /**
     * @brief cast object to child instance
     */
    template <typename T>
    T const* toInstance() const
    {
        static_assert(std::is_base_of<BaseTensorEngineSettings, T>::value, "T should inheritance TensorEngineSettings");
        return m_instance && m_instance->m_instance ? dynamic_cast<T const*>(m_instance.get())
                                                    : dynamic_cast<T const*>(this);
    }

public: // IJsonParsed interface
    bool parse(QJsonObject const& json) override;

public:// ISettings interface
    bool valid() const override;

private:
    BaseTensorEngineSettings(bool makeCommon);

private:
    using TypesConstructors = QMap<QString, TypeConstructor>;
    static TypesConstructors m_typesConstructors;

private:
    std::shared_ptr<BaseTensorEngineSettings> m_instance = nullptr;
};
}
