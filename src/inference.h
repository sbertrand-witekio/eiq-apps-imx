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

#ifndef inference_h
#define inference_h

#include <chrono>
#include <string>
#include <opencv2/core.hpp>
#include <g2d.h>
#include <gst/gst.h>
#include <gst/video/video.h>
extern "C" {
#include "imx_2d_device.h"
}


class inference_t
{
public:

  enum {
    OK = 0,
    ERROR = -1,
  };

  inference_t();
  virtual ~inference_t();

  int init();

  int setup_g2d(void);
  int clean_g2d(void);

  virtual int inference(void) = 0;
  virtual int setup_input_tensor(
    GObject *object,
    GstVideoInfo *vinfo,
    Imx2DFrame *src_frame,
    Imx2DFrame *dst_frame);
  virtual int calc_stats(cv::Mat& frame);
  virtual int draw_stats(cv::Mat& frame);
  virtual int draw_results(cv::Mat& frame) = 0;
  virtual int get_input_tensor_shape(std::vector<int> *shape) = 0;
  virtual int get_input_tensor(uint8_t **ptr, size_t* sz) { return ERROR; }
  virtual int copy_data_to_input_tensor(uint8_t *data, size_t sz) { return ERROR; }

  int setup_g2d_surface(
    GstVideoFormat format,
    int width,
    int height,
    uint8_t *paddr,
    Imx2DRotationMode rotate,
    struct g2d_surface *s);

  double inference_time_cur_ = 0;

  int video_width_ = 0;
  int video_height_ = 0;

  int bgrx_width_ = 0;
  int bgrx_height_ = 0;
  int bgrx_channels_ = 0;

private:

  // g2d for resize
  void *g2d_handle_ = NULL;
  g2d_buf *bgrx_buf_ = NULL;
  int bgrx_stride_ = 0;
  size_t bgrx_size_ = 0;

  // measure fps
  std::chrono::steady_clock::time_point start_time_;
  size_t frame_count_;
  double uptime_;
  double fps_;
  double inference_time_total_;
  double inference_time_avg_;
  // stats
  std::string fps_stats_;
  std::string inference_stats_;
  int stats_initialized_ = 0;

  // unused
  inference_t(const inference_t&);
  inference_t& operator=(const inference_t&);

};

#endif
