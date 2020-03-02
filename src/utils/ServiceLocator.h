#pragma once

#include <QString>
#include <QHash>
#include <functional>

#include <engines/ITensorEngine.h>
#include <image/IImageConvertor.h>


namespace utils
{
class ServiceLocator
{
public:
    void init();

    void setTensorEngineType(QString const& type);

    engines::ITensorEnginePtr createTensorEngine() const;
    image::IImageConvertorPtr createImageConvertor() const;

#ifdef INCLUDE_TENSOR_RT_BUILD
    static constexpr auto TENSOR_RT = "TensorRt";
#endif

private:
   using TensorEngineContructor = std::function<engines::ITensorEnginePtr()>;

private:
    TensorEngineContructor m_tensorEngineContructor{};
    QHash<QString, TensorEngineContructor> m_tensorEngineContructors{};
};
}
