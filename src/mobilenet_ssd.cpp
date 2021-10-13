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

#include "mobilenet_ssd.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <fstream>

GST_DEBUG_CATEGORY(mobilenet_ssd_t_debug);
#define GST_CAT_DEFAULT mobilenet_ssd_t_debug


mobilenet_ssd_t::mobilenet_ssd_t()
{
  GST_DEBUG_CATEGORY_INIT(mobilenet_ssd_t_debug, "mobilenet_ssd_t", 0, "i.MX NN Inference demo TFLite mobilenet_ssd class");
  GST_TRACE("%s", __func__);
}

mobilenet_ssd_t::~mobilenet_ssd_t()
{
  GST_TRACE("%s", __func__);
}

int mobilenet_ssd_t::init(
  const std::string& model,
  int use_nnapi,
  int num_threads)
{
  GST_TRACE("%s", __func__);
  tflite_inference_t::init(model, use_nnapi, num_threads);
  return OK;
}

int
mobilenet_ssd_t::load_labels(
  const std::string& filename)
{
  GST_TRACE("%s", __func__);

  std::ifstream file(filename);
  if (!file) {
    GST_ERROR ("Failed to open %s", filename.c_str());
    return ERROR;
  }
  std::string line;
  while (std::getline(file, line)) {
    std::size_t found = line.find("  ");
    if (found != std::string::npos) {
      std::string id = line.substr(0, found);
      std::string label = line.substr(found + 2);
      std::pair<int, std::string> label_pair(std::stoi(id), label);
      label_.push_back(label_pair);
    }
  }
  return OK;
}

int
mobilenet_ssd_t::get_label(
  int id, std::string& label)
{
  auto it = std::find_if(label_.begin(), label_.end(),
  [&id](const std::pair<int, std::string>& element){ return element.first == id;} );
  if (it != label_.end()) {
    label = it->second;
    return OK;
  }
  return ERROR;
}

int
mobilenet_ssd_t::draw_mobilenet(
  cv::Mat& frame,
  float score,
  const std::string& label,
  float ymin,
  float xmin,
  float ymax,
  float xmax)
{
  cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(10, 255, 0), 4);

  char buf[256];
  if (label.length()) {
    snprintf(buf, 256, "%s: %d%%", label.c_str(), (int)(score * 100));
  }
  else
  {
    snprintf(buf, 256, "unknown: %d%%", (int)(score * 100));
  }

  int baseline = 0;
  cv::Size text_sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);

  int label_ymin = std::max((int)ymin, text_sz.height + 10);  // Make sure not to draw label too close to top of window
  cv::rectangle(frame, cv::Point((int)xmin, label_ymin - text_sz.height - 10), cv::Point((int)xmin + text_sz.width, label_ymin + baseline - 10), cv::Scalar(255, 255, 255), cv::FILLED); // Draw white box to put label text in
  cv::putText(frame, buf, cv::Point((int)xmin, label_ymin - 7), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);// Draw label text

  return OK;
}

int
mobilenet_ssd_t::handle_mobilenet(
  cv::Mat& frame,
  float threshold,
  int image_width,
  int image_height)
{
  size_t sz[4] = {0,};
  float *mn_location = (float *)(typed_output_tensor<float>(0, &sz[0]));
  float *mn_label = (float *)(typed_output_tensor<float>(1, &sz[1]));
  float *mn_score = (float *)(typed_output_tensor<float>(2, &sz[2]));
  float *mn_num_detect = (float *)(typed_output_tensor<float>(3, &sz[3]));

  GST_TRACE("mn results: %p[%ld], %p[%ld], %p[%ld], %p[%ld]",
        mn_location, sz[0],
        mn_label, sz[1],
        mn_score, sz[2],
        mn_num_detect, sz[3]);

  int num_detect = (int)(*mn_num_detect);
  for (int i = 0; i < num_detect; i++) {
    float score = mn_score[i];
    if (score > threshold) {
      int label_id = (int)mn_label[i];
      std::string label_str("unknown");
      get_label(label_id, label_str);

      // Get the bbox, make sure its not out of the image bounds, and scale up to src image size
      float ymin = std::fmax(0.0f, mn_location[4 * i] * image_height);
      float xmin = std::fmax(0.0f, mn_location[4 * i + 1] * image_width);
      float ymax = std::fmin(float(image_height - 1), mn_location[4 * i + 2] * image_height);
      float xmax = std::fmin(float(image_width - 1), mn_location[4 * i + 3] * image_width);

      draw_mobilenet(frame, score, label_str, ymin, xmin, ymax, xmax);
    }
  }
  return OK;
}

int mobilenet_ssd_t::draw_results(cv::Mat& frame)
{
  GST_TRACE("%s", __func__);
  float threshold = 0.49;
  handle_mobilenet(frame, threshold, frame.cols, frame.rows);
  return OK;
}
