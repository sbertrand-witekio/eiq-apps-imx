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

#include "inference.h"
#include "utils.h"
extern "C" {
#include <gst/allocators/gstallocatorphymem.h>
}
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

GST_DEBUG_CATEGORY(inference_t_debug);
#define GST_CAT_DEFAULT inference_t_debug

inference_t::inference_t() :
  bgrx_buf_(NULL),
  g2d_handle_(NULL)
{
  GST_DEBUG_CATEGORY_INIT(inference_t_debug, "inference_t", 0, "i.MX NN Inference demo inference class");
  GST_TRACE("%s", __func__);
}

inference_t::~inference_t()
{
  GST_TRACE("%s", __func__);
  clean_g2d();
}

int inference_t::init()
{
  GST_TRACE("%s", __func__);
  return OK;
}

int inference_t::setup_g2d(void)
{
  GST_TRACE("%s", __func__);

  int ret = 0;
  if (!g2d_handle_) {
    ret = g2d_open(&g2d_handle_);
    if (ret != 0 || g2d_handle_ == NULL)
    {
      GST_ERROR ("g2d_open failed");
      return ERROR;
    }
    GST_TRACE("g2d_handle: %p", g2d_handle_);
  }

  // alloc BGRx buffer
  std::vector<int> shape;
  get_input_tensor_shape(&shape);
  bgrx_height_ = shape[1];
  bgrx_width_ = shape[2];
  bgrx_channels_ = shape[3];
  GST_TRACE("wanted size: %dx%dx%d", bgrx_width_, bgrx_height_, bgrx_channels_);
  bgrx_stride_ = (bgrx_width_ + 15) & (~0xf);
  bgrx_size_ = PAGE_ALIGN(bgrx_stride_ * bgrx_height_ * 4);

  if (!bgrx_buf_) {
    bgrx_buf_ = g2d_alloc(bgrx_size_, 1);
    if (bgrx_buf_ == NULL) {
      GST_ERROR ("g2d_alloc failed");
      return ERROR;
    }
    GST_TRACE("bgrx_buf: %p, p:0x%08x, v:%p", bgrx_buf_, bgrx_buf_->buf_paddr, bgrx_buf_->buf_vaddr);
  }

  return OK;
}

int inference_t::clean_g2d(void)
{
  GST_TRACE("%s", __func__);
  // clean up
  if (bgrx_buf_) {
    g2d_free(bgrx_buf_);
    bgrx_buf_ = NULL;
  }
  if (g2d_handle_) {
    g2d_close(g2d_handle_);
    g2d_handle_ = NULL;
  }
  return OK;
}

int inference_t::setup_input_tensor(
  GObject *object,
  GstVideoInfo *vinfo,
  Imx2DFrame *src_frame,
  Imx2DFrame *dst_frame)
{
  GST_TRACE("%s", __func__);

  int ret = OK;

  // set video size info
  video_width_ = vinfo->width;
  video_height_ = vinfo->height;

  // setup g2d
  ret = setup_g2d();
  if (ret != OK) {
    GST_ERROR("setup_g2d failed");
    return ret;
  }

  // setup src g2d surface
  struct g2d_surface src;
  ret = setup_g2d_surface(
    vinfo->finfo->format,
    vinfo->width,
    vinfo->height,
    src_frame->mem->paddr,
    src_frame->rotate,
    &src);
  if (ret != OK) {
    GST_ERROR("setup_surface failed");
    return ret;
  }

  // setup resized (but aligned for g2d) surface
  struct g2d_surface dst;
  ret = setup_g2d_surface(
    GST_VIDEO_FORMAT_BGRx,
    bgrx_width_,
    bgrx_height_,
    (uint8_t*)(long)(bgrx_buf_->buf_paddr),
    IMX_2D_ROTATION_0,
    &dst);
  if (ret != OK) {
    GST_ERROR("setup_surface failed");
    return ret;
  }

  // blit by g2d api
  ret = g2d_blit(g2d_handle_, &src, &dst);
  if (ret != 0) {
    GST_ERROR ("g2d_blit failed (ret=%d)", ret);
    return ERROR;
  }
  g2d_finish(g2d_handle_);

  // convert BGRx8888 to RGB888
  uint8_t *bgrx = (uint8_t *)bgrx_buf_->buf_vaddr;
  size_t sz = 0;
  uint8_t *rgb = 0;
  ret = get_input_tensor(&rgb, &sz);
  if (ret == OK) {
    GST_TRACE("bgrx, rgb, sz, expected sz = {%p, %p, %ld, %d}", bgrx, rgb, sz, (bgrx_width_ * bgrx_height_ * bgrx_channels_));
    uint8_t *p = bgrx;
    uint8_t *q = rgb;
    utils::bgrx_to_rgb(p, q, bgrx_width_, bgrx_height_, bgrx_stride_);
  } else {
    sz = bgrx_width_ * bgrx_height_ * bgrx_channels_;
    rgb = new uint8_t [sz];
    GST_TRACE("bgrx, rgb, sz, expected sz = {%p, %p, %ld, %d}", bgrx, rgb, sz, (bgrx_width_ * bgrx_height_ * bgrx_channels_));
    uint8_t *p = bgrx;
    uint8_t *q = rgb;
    utils::bgrx_to_rgb(p, q, bgrx_width_, bgrx_height_, bgrx_stride_);
    ret = copy_data_to_input_tensor(q, sz);
    assert(ret == 0);
    delete [] rgb;
  }

  // clean up g2d
  ret = clean_g2d();
  if (ret != OK) {
    GST_ERROR("clean_g2d failed");
    return ERROR;
  }
  return OK;
}

int
inference_t::setup_g2d_surface(
  GstVideoFormat format,
  int width,
  int height,
  uint8_t *paddr,
  Imx2DRotationMode rotate,
  struct g2d_surface *s)
{
  GST_TRACE("%s", __func__);

  s->width = width;
  s->height = height;

  s->global_alpha = 0xff;
  s->left = 0;
  s->top = 0;
  s->right = width;
  s->bottom = height;

  s->stride = (width + 15) & (~0xf); // buffer stride, in Pixels
  s->blendfunc = (g2d_blend_func)0;
  s->clrcolor = 0; // format is RGBA8888, used as dst for clear, as src for blend dim

  // Imx2DRotationMode to enum g2d_rotation
  switch (rotate) {
    case IMX_2D_ROTATION_90:    s->rot = G2D_ROTATION_90; break;
    case IMX_2D_ROTATION_180:   s->rot = G2D_ROTATION_180; break;
    case IMX_2D_ROTATION_270:   s->rot = G2D_ROTATION_270; break;
    case IMX_2D_ROTATION_HFLIP: s->rot = G2D_FLIP_H; break;
    case IMX_2D_ROTATION_VFLIP: s->rot = G2D_FLIP_V; break;
    case IMX_2D_ROTATION_0:
    default:                    s->rot = G2D_ROTATION_0; break;
  }

  // GstVideoFormat to g2d_format
  switch (format) {
    case GST_VIDEO_FORMAT_RGB16: s->format = G2D_RGB565;   break;
    case GST_VIDEO_FORMAT_RGBx:  s->format = G2D_RGBX8888; break;
    case GST_VIDEO_FORMAT_RGBA:  s->format = G2D_RGBA8888; break;
    case GST_VIDEO_FORMAT_BGRA:  s->format = G2D_BGRA8888; break;
    case GST_VIDEO_FORMAT_BGRx:  s->format = G2D_BGRX8888; break;
    case GST_VIDEO_FORMAT_BGR16: s->format = G2D_BGR565;   break;
    case GST_VIDEO_FORMAT_ARGB:  s->format = G2D_ARGB8888; break;
    case GST_VIDEO_FORMAT_ABGR:  s->format = G2D_ABGR8888; break;
    case GST_VIDEO_FORMAT_xRGB:  s->format = G2D_XRGB8888; break;
    case GST_VIDEO_FORMAT_xBGR:  s->format = G2D_XBGR8888; break;
    case GST_VIDEO_FORMAT_I420:  s->format = G2D_I420;     break;
    case GST_VIDEO_FORMAT_NV12:  s->format = G2D_NV12;     break;
    case GST_VIDEO_FORMAT_UYVY:  s->format = G2D_UYVY;     break;
    case GST_VIDEO_FORMAT_YUY2:  s->format = G2D_YUYV;     break;
    case GST_VIDEO_FORMAT_YVYU:  s->format = G2D_YVYU;     break;
    case GST_VIDEO_FORMAT_YV12:  s->format = G2D_YV12;     break;
    case GST_VIDEO_FORMAT_NV16:  s->format = G2D_NV16;     break;
    case GST_VIDEO_FORMAT_NV21:  s->format = G2D_NV21;     break;
    default:
      GST_ERROR ("G2D: not supported format.");
      return ERROR;
  }
  switch (s->format) {
    case G2D_I420:
      s->planes[0] = (gint)(glong)(paddr);
      s->planes[1] = (gint)(glong)(paddr + width * height);
      s->planes[2] = s->planes[1] + width * height / 4;
      break;
    case G2D_YV12:
      s->planes[0] = (gint)(glong)(paddr);
      s->planes[2] = (gint)(glong)(paddr + width * height);
      s->planes[1] = s->planes[2] + width * height / 4;
      break;
    case G2D_NV12:
    case G2D_NV21:
    case G2D_NV16:
      s->planes[0] = (gint)(glong)(paddr);
      s->planes[1] = (gint)(glong)(paddr + width * height);
      s->planes[2] = 0;
      break;
    case G2D_RGB565:
    case G2D_RGBX8888:
    case G2D_RGBA8888:
    case G2D_BGRA8888:
    case G2D_BGRX8888:
    case G2D_BGR565:
    case G2D_ARGB8888:
    case G2D_ABGR8888:
    case G2D_XRGB8888:
    case G2D_XBGR8888:
    case G2D_UYVY:
    case G2D_YUYV:
    case G2D_YVYU:
      s->planes[0] = (gint)(glong)(paddr);
      s->planes[1] = 0;
      s->planes[2] = 0;
      break;
    default:
      GST_ERROR ("G2D: not supported format.");
      return ERROR;
  }
  GST_TRACE(
    "g2d src : %dx%d, %d (%d,%d-%d,%d), alpha=%d, format=%d, planes={x%08x, x%08x, x%08x}",
    s->width, s->height, s->stride,
    s->left, s->top, s->right, s->bottom,
    s->global_alpha, s->format,
    s->planes[0], s->planes[1], s->planes[2]);
  return OK;
}

int inference_t::calc_stats(cv::Mat& frame)
{
  GST_TRACE("%s", __func__);

  if (stats_initialized_) {
    frame_count_++;
    std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
    std::chrono::duration<double> uptime = time - start_time_;
    uptime_ = uptime.count();
    fps_ = (double)frame_count_ / uptime_;
    inference_time_total_ += inference_time_cur_;
    inference_time_avg_ = inference_time_total_ / frame_count_;
  } else {
    // initialize
    start_time_ = std::chrono::steady_clock::now();
    frame_count_ = 0;
    uptime_ = 0.0;
    fps_ = 0;
    stats_initialized_ = 1;
    inference_time_total_ = 0;
    inference_time_avg_ = 0;
  }

  char buf[256];
  // inference time stats
  std::snprintf(
    buf, sizeof(buf),
    "Inference time Avg: %6.3fms, Cur: %6.3fms (%.1ffps)",
    inference_time_avg_,
    inference_time_cur_,
    1000.0f / inference_time_avg_);
  inference_stats_ = buf;
  // fps stats
  std::snprintf(
    buf, sizeof(buf),
    "Video: %6.3ffps (Res: %dx%d, Frame: %ld, Uptime: %.3fs)",
    fps_,
    frame.cols, frame.rows,
    frame_count_,
    uptime_);
  fps_stats_ = buf;

  GST_LOG("%s, %s", inference_stats_.c_str(), fps_stats_.c_str());
  return OK;
}

int inference_t::draw_stats(cv::Mat& frame)
{
  GST_TRACE("%s", __func__);

  // display status text
  int margin_left = 10;
  int margin_bottom = 10;

  float font_scale = 0.7;
  int thickness = 2;

  int baseline = 0;
  cv::Size text_sz_fps = cv::getTextSize(fps_stats_.c_str(), cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
  GST_TRACE("fps-box w,h,b: %d, %d, %d", text_sz_fps.width, text_sz_fps.height, baseline);

  cv::Size text_sz_inf = cv::getTextSize(inference_stats_.c_str(), cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
  GST_TRACE("inf-box w,h,b: %d, %d, %d", text_sz_inf.width, text_sz_inf.height, baseline);

  cv::Size text_sz;
  if (text_sz_fps.width > text_sz_inf.width) {
    text_sz = text_sz_fps;
  } else {
    text_sz = text_sz_inf;
  }

  // Draw white box to put label text in

// this is for performance
#define DRAW_WHITE_RECT

// this is for performance
#define LINE_TYPE (8)
//#define LINE_TYPE (CV_FILLED)
//#define LINE_TYPE (CV_AA)

  cv::Scalar color_white(255, 255, 255);
  cv::Scalar color_black(150, 150, 150);
#if defined(DRAW_WHITE_RECT)
  cv::rectangle(
    frame,
    cv::Point(margin_left - 4, frame.rows - margin_bottom - ((baseline * 2) + (text_sz.height * 2)) - 4),
    cv::Point(margin_left + text_sz.width + 4, frame.rows - margin_bottom + 4),
    color_white,
    cv::FILLED);
#endif
  cv::putText(
    frame,
    inference_stats_,
    cv::Point(margin_left, frame.rows - margin_bottom - (baseline * 2 + text_sz.height)),
    cv::FONT_HERSHEY_SIMPLEX,
    font_scale,
    color_black,
#if defined(DRAW_WHITE_RECT)
    thickness,
#else
    thickness + 1,
#endif
    LINE_TYPE);
#if !defined(DRAW_WHITE_RECT)
  cv::putText(
    frame,
    inference_stats_,
    cv::Point(margin_left, frame.rows - margin_bottom - (baseline * 2 + text_sz.height)),
    cv::FONT_HERSHEY_SIMPLEX,
    font_scale,
    color_white,
    thickness,
    LINE_TYPE);
#endif
  cv::putText(
    frame,
    fps_stats_,
    cv::Point(margin_left, frame.rows - margin_bottom - baseline + 4),
    cv::FONT_HERSHEY_SIMPLEX,
    font_scale,
    color_black,
#if defined(DRAW_WHITE_RECT)
    thickness,
#else
    thickness + 1,
#endif
    LINE_TYPE);
#if !defined(DRAW_WHITE_RECT)
  cv::putText(
    frame,
    fps_stats_,
    cv::Point(margin_left, frame.rows - margin_bottom - baseline + 4),
    cv::FONT_HERSHEY_SIMPLEX,
    font_scale,
    color_white,
    thickness,
    LINE_TYPE);
#endif

  return OK;
}
