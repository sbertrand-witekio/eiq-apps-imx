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

#ifndef gstnninferencedemo_h
#define gstnninferencedemo_h

#include <gst/gst.h>
extern "C" {
#include "gstimxcommon.h"
#include "imx_2d_device.h"
}
#include <chrono>
#include <string>
#include "inference.h"

G_BEGIN_DECLS

/* nninferencedemo object and class definition */
typedef struct _GstNnInferenceDemo {
  GstVideoFilter element;

  Imx2DDevice *device;
  GstBufferPool *in_pool;
  GstBufferPool *out_pool;
  GstBufferPool *self_out_pool;
  GstBuffer *in_buf;
  GstAllocator *allocator;
  GstVideoAlignment in_video_align;
  GstVideoAlignment out_video_align;
  gboolean pool_config_update;

  /* properties */
  enum DemoMode {
    tflite_posenet,
    tflite_mobilenet_ssd,
    tflite_benchmark,
  } demo_mode;
  Imx2DRotationMode rotate;
  gchar *model;
  gchar *label;
  gboolean display_stats;
  gint use_nnapi;
  gboolean enable_inference;
  gint num_threads;

  /* inference object */
  inference_t *inference;
} GstNnInferenceDemo;

typedef struct _GstNnInferenceDemoClass {
  GstVideoFilterClass parent_class;
  const Imx2DDeviceInfo *in_plugin;
} GstNnInferenceDemoClass;

G_END_DECLS

#endif /* gstnninferencedemo_h */
