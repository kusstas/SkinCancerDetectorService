#include "ServiceLocator.h"

#include "image/opencv/ImageConvertor.h"

#ifdef INCLUDE_TENSOR_RT_BUILD
#include "engines/tensorRt/TensorEngine.h"
#include "engines/tensorRt/TensorEngineSettings.h"
#endif

#ifdef INCLUDE_TORCH_BUILD
#include "engines/torch/TensorEngine.h"
#include "engines/torch/TensorEngineSettings.h"
#endif

#include <QMetaType>


namespace utils
{
void ServiceLocator::init()
{
    qRegisterMetaType<common::IEngineInputDataPtr>("IEngineInputDataPtr");

#ifdef INCLUDE_TENSOR_RT_BUILD
    engines::tensorRt::TensorEngineSettings::registerSelf(TENSOR_RT);
    m_tensorEngineContructors.insert(TENSOR_RT, [] () { return std::make_shared<engines::tensorRt::TensorEngine>(); });
#endif

#ifdef INCLUDE_TORCH_BUILD
    engines::torch::TensorEngineSettings::registerSelf(TORCH);
    m_tensorEngineContructors.insert(TORCH, [] () { return std::make_shared<engines::torch::TensorEngine>(); });
#endif
}

void ServiceLocator::setTensorEngineType(QString const& type)
{
    m_tensorEngineContructor = m_tensorEngineContructors.value(type, {});
}

engines::ITensorEnginePtr ServiceLocator::createTensorEngine() const
{
    if (m_tensorEngineContructor)
    {
        return m_tensorEngineContructor();
    }

    return nullptr;
}

image::IImageConvertorPtr ServiceLocator::createImageConvertor() const
{
    return std::make_shared<image::opencv::ImageConvertor>();
}
}
