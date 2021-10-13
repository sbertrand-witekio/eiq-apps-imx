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

#ifndef mobilenet_ssd_h
#define mobilenet_ssd_h

#include "tflite_inference.h"

class mobilenet_ssd_t : public tflite_inference_t
{
public:

  enum {
    OK = 0,
    ERROR = -1,
  };

  mobilenet_ssd_t();
  virtual ~mobilenet_ssd_t();

  int init(
    const std::string& model,
    int use_nnapi = 2,
    int num_threads = 4);

  virtual int load_labels(
    const std::string& label);

  virtual int draw_results(cv::Mat& frame);

  int get_label(int id, std::string& label);

  std::vector<std::pair<int, std::string>> label_;


  int draw_mobilenet(
    cv::Mat& frame,
    float score,
    const std::string& label,
    float ymin,
    float xmin,
    float ymax,
    float xmax);

  int handle_mobilenet(
    cv::Mat& frame,
    float threshold,
    int image_width,
    int image_height);

private:

  // unused
  mobilenet_ssd_t(const mobilenet_ssd_t&);
  mobilenet_ssd_t& operator=(const mobilenet_ssd_t&);

};

#endif
