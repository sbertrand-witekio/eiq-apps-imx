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

#ifndef tflite_inference_h
#define tflite_inference_h

#include "tensorflow/lite/kernels/register.h"
#include "inference.h"

class tflite_inference_t : public inference_t
{
public:

  enum {
    OK = 0,
    ERROR = -1,
  };

  tflite_inference_t();
  virtual ~tflite_inference_t();

  int init(
    const std::string& model,
    int use_nnapi,
    int num_threads);

  virtual int inference(void);
  virtual int get_input_tensor_shape(std::vector<int>* shape);
  virtual int get_input_tensor(uint8_t **ptr, size_t* sz);

  bool verbose_ = false;

protected:

  template <typename T>
  T* typed_tensor(
    int index,
    size_t* length = NULL)
  {
    if (length) {
      *length = 0;
      TfLiteIntArray *dims = interpreter_->tensor(index)->dims;
      if (dims && dims->size && dims->data[0]) {
        *length = 1;
        for (int i = 0; i < dims->size; i++) {
          *length *= dims->data[i];
        }
      }
    }
    return interpreter_->typed_tensor<T>(index);
  }

  template <typename T>
  const T* typed_tensor(
    int index,
    size_t* length = NULL) const
  {
    if (length) {
      *length = 0;
      TfLiteIntArray *dims = interpreter_->tensor(index)->dims;
      if (dims && dims->size && dims->data[0]) {
        *length = 1;
        for (int i = 0; i < dims->size; i++) {
          *length *= dims->data[i];
        }
      }
    }
    return interpreter_->typed_tensor<T>(index);
  }

  template <typename T>
  T* typed_input_tensor(
    int index,
    size_t* length = NULL)
  {
    return typed_tensor<T>(interpreter_->inputs()[index], length);
  }

  template <typename T>
  const T* typed_input_tensor(
    int index,
    size_t* length = NULL) const
  {
    return typed_tensor<T>(interpreter_->inputs()[index], length);
  }

  template <typename T>
  T* typed_output_tensor(
    int index,
    size_t* length = NULL)
  {
    return typed_tensor<T>(interpreter_->outputs()[index], length);
  }

  template <typename T>
  const T* typed_output_tensor(
    int index,
    size_t* length = NULL) const
  {
    return typed_tensor<T>(interpreter_->outputs()[index], length);
  }

  const std::vector<int>& inputs() const
  {
    return interpreter_->inputs();
  }

  const char* get_input_name(int index) const
  {
    return interpreter_->tensor(interpreter_->inputs()[index])->name;
  }

  const std::vector<int>& outputs() const
  {
    return interpreter_->outputs();
  }

  const char* get_output_name(int index) const
  {
    return interpreter_->tensor(interpreter_->outputs()[index])->name;
  }

  std::unique_ptr<tflite::Interpreter> interpreter_;

private:

  int apply_delegate(
    int use_nnapi);

  std::unique_ptr<tflite::FlatBufferModel> model_;

  // unused
  tflite_inference_t(const tflite_inference_t&);
  tflite_inference_t& operator=(const tflite_inference_t&);

};

#endif
