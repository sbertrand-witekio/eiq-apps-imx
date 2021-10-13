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

#ifndef tflite_benchmark_h
#define tflite_benchmark_h

#include "tflite_inference.h"

class tflite_benchmark_t : public tflite_inference_t
{
public:

  enum {
    OK = 0,
    ERROR = -1,
  };

  tflite_benchmark_t();

  virtual ~tflite_benchmark_t();

  int init(
    const std::string& model,
    int use_nnapi = 2,
    int num_threads = 4);

  int draw_results(cv::Mat& frame);

private:

  // unused
  tflite_benchmark_t(const tflite_benchmark_t&);
  tflite_benchmark_t& operator=(const tflite_benchmark_t&);

};

#endif
