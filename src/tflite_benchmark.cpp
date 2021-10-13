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

#include "tflite_benchmark.h"

GST_DEBUG_CATEGORY(tflite_benchmark_t_debug);
#define GST_CAT_DEFAULT tflite_benchmark_t_debug


tflite_benchmark_t::tflite_benchmark_t()
{
  GST_DEBUG_CATEGORY_INIT(tflite_benchmark_t_debug, "tflite_benchmark_t", 0, "i.MX NN Inference demo TFLite benchmark class");
  GST_TRACE("%s", __func__);
}

tflite_benchmark_t::~tflite_benchmark_t()
{
  GST_TRACE("%s", __func__);
}

int tflite_benchmark_t::init(
  const std::string& model,
  int use_nnapi,
  int num_threads)
{
  GST_TRACE("%s", __func__);
  return tflite_inference_t::init(model, use_nnapi, num_threads);
}

int tflite_benchmark_t::draw_results(cv::Mat& frame)
{
  GST_TRACE("%s", __func__);
  return OK;
}
