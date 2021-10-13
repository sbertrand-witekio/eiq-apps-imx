/* GStreamer i.MX NN Inference demo plugin
 *
 * Copyright 2021 NXP
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tflite_inference.h"

// google-coral/edgetpu
#include "posenet/posenet_decoder_op.h"
#ifdef BUILD_WITH_EDGETPU
#include "edgetpu.h"
#endif

// tensorflow/lite
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>

// std
#include <map>
#include <fstream>


GST_DEBUG_CATEGORY(tflite_inference_t_debug);
#define GST_CAT_DEFAULT tflite_inference_t_debug

tflite_inference_t::tflite_inference_t()
{
  GST_DEBUG_CATEGORY_INIT(tflite_inference_t_debug, "tflite_inference_t", 0, "i.MX NN Inference demo tflite_inference class");
  GST_TRACE("%s", __func__);
}

tflite_inference_t::~tflite_inference_t()
{
  GST_TRACE("%s", __func__);

  model_.reset();
  interpreter_.reset();
}

int tflite_inference_t::init(
  const std::string& model,
  int use_nnapi,
  int num_threads)
{
  GST_TRACE("%s", __func__);

  // check model existence
  std::ifstream file(model);
  if (!file) {
    GST_ERROR ("Failed to open %s", model.c_str());
    return ERROR;
  }

  model_ = tflite::FlatBufferModel::BuildFromFile(model.c_str());
  if (!model_) {
    GST_ERROR ("Failed to mmap model %s", model.c_str());
    return ERROR;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());
#ifdef BUILD_WITH_EDGETPU
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
#endif

  tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
  if (!interpreter_) {
    GST_ERROR ("Failed to construct TFLite interpreter");
    return ERROR;
  }
  bool allow_fp16 = false;
  interpreter_->SetAllowFp16PrecisionForFp32(allow_fp16);
#ifdef BUILD_WITH_EDGETPU
  // Bind edgeTpu context with interpreter.
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext, (TfLiteExternalContext*)edgetpu_context.get());
  interpreter_->SetNumThreads(1);// num_of_thread is ignored
#else
  interpreter_->SetNumThreads(num_threads);
#endif

  apply_delegate(use_nnapi);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    GST_ERROR ("Failed to allocate TFLite tensors!");
    return ERROR;
  }

  if (verbose_) {
    tflite::PrintInterpreterState(interpreter_.get());
  }

  // initial inference test
  int width = 0;
  int height = 0;
  int channel = 0;
  std::vector<int> shape;
  get_input_tensor_shape(&shape);
  height = shape[1];
  width = shape[2];
  channel = shape[3];
  GST_TRACE("input shape: %dx%dx%d", width, height, channel);
  if ((width <= 0) || (height <= 0) || (channel != 3)) {
    GST_ERROR("Not supported input shape");
    return ERROR;
  }
  size_t sz = 0;
  uint8_t* p = 0;
  int ret = get_input_tensor(&p, &sz);
  std::memset(p, 0, sz);
  if (interpreter_->Invoke() != kTfLiteOk) {
    GST_ERROR("Failed to invoke TFLite interpreter");
    return ERROR;
  }

  return OK;
}

int tflite_inference_t::apply_delegate(
  int use_nnapi)
{
  GST_TRACE("%s", __func__);

  // assume TFLite v2.0 or newer
  std::map<std::string, tflite::Interpreter::TfLiteDelegatePtr> delegates;
  if (use_nnapi == 1) {
    auto delegate = tflite::Interpreter::TfLiteDelegatePtr(tflite::NnApiDelegate(), [](TfLiteDelegate*) {});
    if (!delegate) {
      GST_WARNING("NNAPI acceleration is unsupported on this platform.");
    } else {
      delegates.emplace("NNAPI", std::move(delegate));
    }
  } else if (use_nnapi == 2) {
    auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
    auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    auto delegate = tflite::Interpreter::TfLiteDelegatePtr(ext_delegate_ptr, [](TfLiteDelegate*) {});
    if (!delegate) {
      GST_WARNING("vx-delegate backend is unsupported on this platform.");
    } else {
      delegates.emplace("vx-delegate", std::move(delegate));
    }
  }

  for (const auto& delegate : delegates) {
    if (interpreter_->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk) {
      GST_ERROR("Failed to apply %s delegate.", delegate.first.c_str());
      return ERROR;
    } else {
      if (verbose_)
      {
        GST_INFO("Applied %s delegate.", delegate.first.c_str());
      }
    }
  }
  return OK;
}

int tflite_inference_t::inference(void)
{
  GST_TRACE("%s", __func__);

  std::chrono::steady_clock::time_point inference_start = std::chrono::steady_clock::now();

  // tflite inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    return ERROR;
  }

  std::chrono::steady_clock::time_point inference_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> inference_time = inference_end - inference_start;
  inference_time_cur_ = std::chrono::duration_cast<std::chrono::nanoseconds>(inference_time).count() / 1000000.0;

  return OK;
}

int tflite_inference_t::get_input_tensor_shape(
  std::vector<int> *shape)
{
  GST_TRACE("%s", __func__);

  shape->clear();
  TfLiteIntArray *dims = interpreter_->tensor(interpreter_->inputs()[0])->dims;
  if (dims) {
    for (int i = 0; i < dims->size; i++) {
      shape->push_back(dims->data[i]);
    }
  }
  return OK;
}

int tflite_inference_t::get_input_tensor(
  uint8_t **ptr,
  size_t* sz)
{
  GST_TRACE("%s", __func__);

  *ptr = typed_input_tensor<uint8_t>(0, sz);
  return OK;
}
