#pragma once

#include <QLoggingCategory>
#include <QStringBuilder>
#include <QJsonObject>
#include <QJsonValue>
#include <QJsonArray>
#include <QList>
#include <QVector>
#include <QString>
#include <climits>
#include <type_traits>
#include <cxxabi.h>

#include "common/IJsonParsed.h"


namespace utils
{
class JsonHelper
{
public:
    using LogCategory =  QLoggingCategory const& (*)();


    inline static QLoggingCategory const& defaultLogCategory()
    {
        return *QLoggingCategory::defaultCategory();
    }

    JsonHelper(LogCategory const& logCategory = defaultLogCategory)
        : m_logCategory(logCategory)
    {
    }

    template <typename T>
    inline bool get(QJsonObject const& obj, QString const& key, T& out, bool warning = false) const;
    template <typename T>
    inline bool get(QJsonObject const& obj, char const* key, T& out, bool warning = false) const;
    template <typename T>
    inline bool get(QJsonObject const& obj, T& out, bool warning = false) const;

    template <typename T>
    inline bool getArray(QJsonObject const& obj, QString const& key, T& out, bool warning = false) const;
    template <typename T>
    inline bool getArray(QJsonObject const& obj, char const* key, T& out, bool warning = false) const;
    template <typename T>
    inline bool getArray(QJsonObject const& obj, T& out, bool warning = false) const;

    inline bool get(QJsonObject const& obj, QString const& key, common::IJsonParsed* out, bool warning = false) const;
    inline bool get(QJsonObject const& obj, char const* key, common::IJsonParsed* out, bool warning = false) const;
    inline bool get(QJsonObject const& obj, common::IJsonParsed* out, bool warning = false) const;

private:
    static constexpr auto KEY_DELIMETER = "=>";
    static constexpr auto KEY_ARRAY = "[%1]";

    using Validator = bool (QJsonValue::*)() const;

    struct Context
    {
        QString key{};
        bool warning = false;

        void append(QString const& key)
        {
            this->key = this->key.isEmpty() ? key : this->key % KEY_DELIMETER % key;
        }
    };

private:
    template <typename T, typename C, typename... Args>
    inline bool get(Context const& ctx,
             QJsonValue const& value,
             T& out,
             Validator const& validator,
             C const& caster,
             Args const&... args) const;
    template <typename T>
    inline bool get(Context const& ctx, QJsonValue const& value, T& out) const;
    template <typename T>
    inline bool getNumber(Context const& ctx, QJsonValue const& value, T& out) const;
    template <typename T>
    inline bool getArray(Context const& ctx, QJsonValue const& value, T& out) const;

private:
    LogCategory m_logCategory = nullptr;
};

template <typename T, typename C, typename... Args>
inline bool JsonHelper::get(Context const& ctx,
                     QJsonValue const& value,
                     T& out,
                     Validator const& validator,
                     C const& caster,
                     Args const&... args) const
{
    if ((value.*validator)())
    {
        out = static_cast<T>((value.*caster)(args...));
        return true;
    }

    if (ctx.warning)
    {
        qCWarning(m_logCategory) << "Cannot assign:" << value << "from:" << ctx.key << "to" << typeid(T).name();
    }

    return false;
}

template <typename T>
inline bool JsonHelper::getNumber(Context const& ctx, QJsonValue const& value, T& out) const
{
    double tmp = out;
    if (get(ctx, value, tmp))
    {
        if (tmp >= static_cast<double>(std::numeric_limits<T>::min())
                && tmp <= static_cast<double>(std::numeric_limits<T>::max())
                && qFuzzyCompare(tmp, static_cast<double>(static_cast<T>(tmp))))
        {
            out = static_cast<T>(tmp);
            return true;
        }
        else
        {
            if (ctx.warning)
            {
                auto typeName = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
                qCWarning(m_logCategory) << "Cannot assign number:"
                                         << value
                                         << "from:"
                                         << ctx.key
                                         << "to"
                                         << typeName;
                delete typeName;
            }
        }
    }

    return false;
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, bool& out) const
{
    return get(ctx, value, out, &QJsonValue::isBool, &QJsonValue::toBool, out);
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, float& out) const
{
    double tmp = out;
    return getNumber(ctx, value, tmp) && (out = static_cast<float>(tmp), true);
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, double& out) const
{
    return get(ctx, value, out, &QJsonValue::isDouble, &QJsonValue::toDouble, out);
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, int& out) const
{
    return getNumber(ctx, value, out);
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, size_t& out) const
{
    return getNumber(ctx, value, out);
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, QString& out) const
{
    return get(ctx, value, out, &QJsonValue::isString, qOverload<>(&QJsonValue::toString));
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, QJsonObject& out) const
{
    return get(ctx, value, out, &QJsonValue::isObject, qOverload<>(&QJsonValue::toObject));
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, QJsonArray& out) const
{
    return get(ctx, value, out, &QJsonValue::isArray, qOverload<>(&QJsonValue::toArray));
}

template <>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, common::IJsonParsed& out) const
{
    QJsonObject obj;
    return get(ctx, value, obj) && out.parse(obj);
}

template <typename T>
inline bool JsonHelper::get(QJsonObject const& obj, QString const& key, T& out, bool warning) const
{
    Context ctx;
    ctx.key = key;
    ctx.warning = warning;

    if (obj.isEmpty())
    {
        if (warning)
        {
            qCWarning(m_logCategory) << "Cannot get:" << key << "from:" << obj;
        }
        return false;
    }

    return get(ctx, obj[key], out);
}

template <typename T>
inline bool JsonHelper::get(QJsonObject const& obj, char const* key, T& out, bool warning) const
{
    return get<T>(obj, QString(key), out, warning);
}

template <typename T>
inline bool JsonHelper::get(QJsonObject const& obj, T& out, bool warning) const
{
    Context ctx;
    ctx.warning = warning;

    return get(ctx, obj, out);
}

template <typename T>
inline bool JsonHelper::getArray(QJsonObject const& obj, QString const& key, T& out, bool warning) const
{
    Context ctx;
    ctx.key = key;
    ctx.warning = warning;

    if (obj.isEmpty())
    {
        if (warning)
        {
            qCWarning(m_logCategory) << "Cannot access to array:" << key << "from:" << obj;
        }
        return false;
    }

    return getArray(ctx, obj[key], out);
}

template <typename T>
inline bool JsonHelper::getArray(QJsonObject const& obj, char const* key, T& out, bool warning) const
{
    return getArray<T>(obj, QString(key), out, warning);
}

template <typename T>
inline bool JsonHelper::getArray(QJsonObject const& obj, T& out, bool warning) const
{
    Context ctx;
    ctx.warning = warning;

    return getArray(ctx, obj, out);
}

inline bool JsonHelper::get(QJsonObject const& obj, QString const& key, common::IJsonParsed* out, bool warning) const
{
    return get(obj, key, *out, warning);
}

inline bool JsonHelper::get(QJsonObject const& obj, char const* key, common::IJsonParsed* out, bool warning) const
{
    return get(obj, key, *out, warning);
}

inline bool JsonHelper::get(QJsonObject const& obj, common::IJsonParsed* out, bool warning) const
{
    return get(obj, *out, warning);
}


template <typename T>
inline bool JsonHelper::get(Context const& ctx, QJsonValue const& value, T& out) const
{
    Q_UNUSED(out);

    if (ctx.warning)
    {
        auto typeName = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
        qCWarning(m_logCategory) << "Cannot assign value:"
                                 << value
                                 << "from:"
                                 << ctx.key
                                 << "to"
                                 << typeName;
        delete typeName;
    }

    return false;
}

template <typename T>
inline bool JsonHelper::getArray(Context const& ctx, QJsonValue const& value, T& out) const
{
    QJsonArray jsonArray;

    if (get(ctx, value, jsonArray))
    {
        T tmp;

        int i = 0;
        for (auto const& elem : jsonArray)
        {
            Context elemCtx = ctx;
            elemCtx.append(QString(KEY_ARRAY).arg(i++));

            tmp.push_back(typename T::value_type());
            auto& outValue = tmp.last();
            if (!get(elemCtx, elem, outValue))
            {
                return false;
            }
        }

        out = std::move(tmp);
        return true;
    }

    return false;
}
}
