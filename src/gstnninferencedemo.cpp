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

#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/allocators/gstdmabufmeta.h>
#include <libdrm/drm_fourcc.h>
extern "C" {
#include <gst/allocators/gstallocatorphymem.h>
}
#include <gst/allocators/gstphymemmeta.h>

#include "gstnninferencedemo.h"
#include "tflite_benchmark.h"
#include "posenet.h"
#include "mobilenet_ssd.h"

#define IN_POOL_MAX_BUFFERS (30)

#define PARAMS_QDATA g_quark_from_static_string("nninferencedemo-params")

#define ROTATION_DEFAULT (IMX_2D_ROTATION_0)
#define DEMO_MODE_DEFAULT (GstNnInferenceDemo::tflite_posenet)
#define DISPLAY_STATS_DEFAULT (TRUE)
#define ENABLE_INFERENCE_DEFAULT (TRUE)
#define USE_NNAPI_DEFAULT (2)
#define NUM_THREADS_DEFAULT (4)
#define MODEL_DEFAULT ""
#define LABEL_DEFAULT ""

#define SHARED_DIR "/usr/share/gstnninferencedemo/"
#define DEFAULT_MODEL_POSENET       SHARED_DIR "google-coral/project-posenet/posenet_mobilenet_v1_075_353_481_quant_decoder.tflite"
#define DEFAULT_MODEL_MOBILENET_SSD SHARED_DIR "google-coral/examples-camera/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
#define DEFAULT_LABEL_MOBILENET_SSD SHARED_DIR "google-coral/examples-camera/coco_labels.txt"

#define UNREF_BUFFER(buffer) {                \
  if (buffer) {                               \
    GST_LOG ("unref buffer (%p)", buffer);    \
    gst_buffer_unref (buffer);                \
    buffer = NULL;                            \
  }                                           \
}

#define UNREF_POOL(pool)  {                   \
  if (pool) {                                 \
    GST_LOG ("unref pool (%p)", pool);        \
    gst_buffer_pool_set_active (pool, FALSE); \
    gst_object_unref (pool);                  \
    pool = NULL;                              \
  }                                           \
}

/* properties utility */
enum {
  PROP_0,
  PROP_OUTPUT_ROTATE,
  PROP_DEMO_MODE,
  PROP_MODEL,
  PROP_LABEL,
  PROP_DISPLAY_STATS,
  PROP_ENABLE_INFERENCE,
  PROP_USE_NNAPI,
  PROP_NUM_THREADS
};

static GstElementClass *parent_class = NULL;

GST_DEBUG_CATEGORY (nninferencedemo_debug);
#define GST_CAT_DEFAULT nninferencedemo_debug


static int
nninferencedemo_init (
  GstNnInferenceDemo * demo)
{
  int ret = 0;
  demo->inference = NULL;
  switch (demo->demo_mode) {
    case GstNnInferenceDemo::tflite_posenet: {
      posenet_t *inference = new posenet_t ();
      std::string model (DEFAULT_MODEL_POSENET);
      if (demo->model)
      {
        model = demo->model;
      }
      ret = inference->init (model, demo->use_nnapi, demo->num_threads);
      demo->inference = inference;
      break;
    }
    case GstNnInferenceDemo::tflite_mobilenet_ssd: {
      mobilenet_ssd_t *inference = new mobilenet_ssd_t ();
      std::string model (DEFAULT_MODEL_MOBILENET_SSD);
      if (demo->model) {
        model = demo->model;
      }
      ret = inference->init (model, demo->use_nnapi, demo->num_threads);
      if (ret == 0) {
        std::string label (DEFAULT_LABEL_MOBILENET_SSD);
        if (demo->label) {
          ret = inference->load_labels (demo->label);
        }
        else {
          ret = inference->load_labels (label);
        }
      }
      demo->inference = inference;
      break;
    }
    case GstNnInferenceDemo::tflite_benchmark: {
      tflite_benchmark_t *inference = new tflite_benchmark_t ();
      if (!demo->model) {
        GST_ERROR ("invalid model");
        return -1;
      }
      std::string model = demo->model;
      ret = inference->init (model, demo->use_nnapi, demo->num_threads);
      demo->inference = inference;
      break;
    }
    default:
      GST_ERROR ("Invalid demo_mode");
      return -1;
  }

  if (ret != 0) {
    GST_ERROR ("Failed to init NN Inference demo");
    return -1;
  }
  return 0;
}

static int nninference (
  GObject *object,
  GstVideoInfo *vinfo,
  Imx2DFrame *src_frame,
  Imx2DFrame *dst_frame)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *) object;
  int ret = 0;
  cv::Mat frameBGRX (vinfo->height, vinfo->width, CV_8UC4, dst_frame->mem->vaddr);
  if (demo->inference) {
    if (demo->enable_inference) {
      ret = demo->inference->setup_input_tensor (object, vinfo, src_frame, dst_frame);
      ret = demo->inference->inference ();
      ret = demo->inference->draw_results (frameBGRX);
    }
    ret = demo->inference->calc_stats (frameBGRX);
    if (demo->display_stats) {
      ret = demo->inference->draw_stats (frameBGRX);
    }
  }
  //GST_TRACE("dst_frame: %d,%d,%d", dst_frame->info.w, dst_frame->info.h, dst_frame->info.stride);
  return 0;
}

static GType
rotation_get_type (void)
{
  static GType rotation_type = 0;

  if (!rotation_type) {
    static GEnumValue rotation_values[] = {
      {IMX_2D_ROTATION_0,     "No rotation",        "none"},
      {IMX_2D_ROTATION_90,    "Rotate 90 degrees",  "rotate-90"},
      {IMX_2D_ROTATION_180,   "Rotate 180 degrees", "rotate-180"},
      {IMX_2D_ROTATION_270,   "Rotate 270 degrees", "rotate-270"},
      {IMX_2D_ROTATION_HFLIP, "Flip horizontally",  "horizontal-flip"},
      {IMX_2D_ROTATION_VFLIP, "Flip vertically",    "vertical-flip"},
      {0,                     NULL,                 NULL },
    };

    rotation_type =
      g_enum_register_static("RotationMode", rotation_values);
  }

  return rotation_type;
}

static GType
deinterlace_get_type (void)
{
  static GType deinterlace_type = 0;

  if (!deinterlace_type) {
    static GEnumValue deinterlace_values[] = {
      {IMX_2D_DEINTERLACE_NONE,        "No deinterlace",            "none"},
      {IMX_2D_DEINTERLACE_LOW_MOTION,  "low-motion deinterlace",    "low-motion"},
      {IMX_2D_DEINTERLACE_MID_MOTION,  "midium-motion deinterlace", "mid-motion"},
      {IMX_2D_DEINTERLACE_HIGH_MOTION, "high-motion deinterlace",   "high-motion"},
      {0,                              NULL,                        NULL},
    };

    deinterlace_type =
      g_enum_register_static("DeinterlaceMode", deinterlace_values);
  }

  return deinterlace_type;
}

static GType
demo_mode_get_type (void)
{
  static GType demo_mode_type = 0;

  if (!demo_mode_type) {
    static GEnumValue demo_mode_values[] = {
      {GstNnInferenceDemo::tflite_posenet,       "TensorFlow Lite Posenet",       "posenet"},
      {GstNnInferenceDemo::tflite_mobilenet_ssd, "TensorFlow Lite Mobilenet SSD", "mobilenet-ssd"},
      {GstNnInferenceDemo::tflite_benchmark,     "TensorFlow Lite Benchmark",     "benchmark"},
      {0,                                        NULL,                            NULL },
    };

    demo_mode_type =
      g_enum_register_static("DemoMode", demo_mode_values);
  }

  return demo_mode_type;
}

static void
set_property (
  GObject * object,
  guint prop_id,
  const GValue * value,
  GParamSpec * pspec)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *) (object);
  Imx2DDevice *device = demo->device;

  GST_DEBUG ("set_property (%d).", prop_id);

  if (!device)
    return;

  switch (prop_id) {
    case PROP_OUTPUT_ROTATE:
      demo->rotate = (Imx2DRotationMode)g_value_get_enum (value);
      break;
    case PROP_DEMO_MODE:
      demo->demo_mode = (GstNnInferenceDemo::DemoMode)g_value_get_enum (value);
      break;
    case PROP_MODEL:
      g_free (demo->model);
      demo->model = g_value_dup_string (value);
      break;
    case PROP_LABEL:
      g_free (demo->label);
      demo->label = g_value_dup_string (value);
      break;
    case PROP_DISPLAY_STATS:
      demo->display_stats = g_value_get_boolean (value);
      break;
    case PROP_ENABLE_INFERENCE:
      demo->enable_inference = g_value_get_boolean (value);
      break;
    case PROP_USE_NNAPI:
      demo->use_nnapi = g_value_get_int (value);
      break;
    case PROP_NUM_THREADS:
      demo->num_threads = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  //TODO if property changed, it may affect the passthrough, so we need
  // reconfig the pipeline, send a reconfig event for caps re-negotiation.
}

static void
get_property (
  GObject * object,
  guint prop_id,
  GValue * value,
  GParamSpec * pspec)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *) (object);
  Imx2DDevice *device = demo->device;

  if (!device)
    return;

  switch (prop_id) {
    case PROP_OUTPUT_ROTATE:
      g_value_set_enum (value, demo->rotate);
      break;
    case PROP_DEMO_MODE:
      g_value_set_enum (value, demo->demo_mode);
      break;
    case PROP_MODEL:
      g_value_set_string (value, demo->model);
      break;
    case PROP_LABEL:
      g_value_set_string (value, demo->label);
      break;
    case PROP_DISPLAY_STATS:
      g_value_set_boolean (value, demo->display_stats);
      break;
    case PROP_ENABLE_INFERENCE:
      g_value_set_boolean (value, demo->enable_inference);
      break;
    case PROP_USE_NNAPI:
      g_value_set_int (value, demo->use_nnapi);
      break;
    case PROP_NUM_THREADS:
      g_value_set_int (value, demo->num_threads);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
finalize (
  GObject * object)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *) (object);
  GstStructure *config;
  GstNnInferenceDemoClass *klass =
        (GstNnInferenceDemoClass *) G_OBJECT_GET_CLASS (demo);
  UNREF_BUFFER (demo->in_buf);
  UNREF_POOL (demo->in_pool);
  UNREF_POOL (demo->self_out_pool);
  if (demo->allocator) {
    gst_object_unref (demo->allocator);
    demo->allocator = NULL;
  }

  if (demo->device) {
    demo->device->close (demo->device);
    if (klass->in_plugin)
      klass->in_plugin->destroy (demo->device);
    demo->device = NULL;
  }

  g_free (demo->model);
  g_free (demo->label);

  if (demo->inference) {
    delete demo->inference;
    demo->inference = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize ((GObject *) (demo));
}

static gboolean
src_event (
  GstBaseTransform * transform,
  GstEvent * event)
{
  gdouble a;
  GstStructure *structure;
  GstVideoFilter *filter = GST_VIDEO_FILTER_CAST (transform);

  GST_TRACE ("%s event", GST_EVENT_TYPE_NAME (event));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_NAVIGATION:
      if ((filter->in_info.width != filter->out_info.width) ||
          (filter->in_info.height != filter->out_info.height)) {
        event = GST_EVENT (gst_mini_object_make_writable(GST_MINI_OBJECT(event)));

        structure = (GstStructure *) gst_event_get_structure (event);
        if (gst_structure_get_double (structure, "pointer_x", &a)) {
          gst_structure_set(
            structure, "pointer_x", G_TYPE_DOUBLE,
            a * filter->in_info.width / filter->out_info.width,
            NULL
          );
        }

        if (gst_structure_get_double (structure, "pointer_y", &a)) {
          gst_structure_set (
            structure, "pointer_y", G_TYPE_DOUBLE,
            a * filter->in_info.height / filter->out_info.height,
            NULL
          );
        }
      }
      break;
    default:
      break;
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->src_event(transform, event);
}

static GstCaps *
transform_caps(
  GstBaseTransform * transform,
  GstPadDirection direction,
  GstCaps * caps,
  GstCaps * filter)
{
  GstCaps *tmp, *tmp2, *result;
  GstStructure *st;
  gint i, n;

  GST_DEBUG ("transform caps: %" GST_PTR_FORMAT, caps);
  GST_DEBUG ("filter: %" GST_PTR_FORMAT, filter);
  GST_DEBUG ("direction: %d", direction);

  /* Get all possible caps that we can transform to */
  /* copies the given caps */
  tmp = gst_caps_new_empty ();
  n = gst_caps_get_size (caps);

  for (i = 0; i < n; i++) {
    st = gst_caps_get_structure (caps, i);

    if ((i > 0) && gst_caps_is_subset_structure (tmp, st))
      continue;

    st = gst_structure_copy (st);

    /* NV12 8x8 to YUY2 works on DPU */
    if (HAS_DPU ()) {
      gst_structure_set (st, "width", GST_TYPE_INT_RANGE, 8, G_MAXINT32,
          "height", GST_TYPE_INT_RANGE, 8, G_MAXINT32, NULL);
    } else {
      gst_structure_set (st, "width", GST_TYPE_INT_RANGE, 64, G_MAXINT32,
          "height", GST_TYPE_INT_RANGE, 64, G_MAXINT32, NULL);
    }

    gst_structure_remove_fields (st, "format", NULL);

    /* if pixel aspect ratio, make a range of it*/
    if (gst_structure_has_field (st, "pixel-aspect-ratio")) {
      gst_structure_set (st, "pixel-aspect-ratio",
          GST_TYPE_FRACTION_RANGE, 1, G_MAXINT32, G_MAXINT32, 1, NULL);
    }

    gst_caps_append_structure (tmp, st);
  }

  if (filter) {
    tmp2 = gst_caps_intersect_full(filter, tmp, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref(tmp);
    tmp = tmp2;
  }

  result = tmp;

  GST_DEBUG("return caps: %" GST_PTR_FORMAT, result);

  return result;
}

/* calculate how much loss a conversion would be */
/* This loss calculation comes from gstvideoconvert.c of base plugins */
static gint
get_format_conversion_loss(
  GstBaseTransform * base,
  GstVideoFormat in_name,
  GstVideoFormat out_name)
{
#define SCORE_FORMAT_CHANGE       1
#define SCORE_DEPTH_CHANGE        1
#define SCORE_ALPHA_CHANGE        1
#define SCORE_CHROMA_W_CHANGE     1
#define SCORE_CHROMA_H_CHANGE     1
#define SCORE_PALETTE_CHANGE      1

#define SCORE_COLORSPACE_LOSS     2     /* RGB <-> YUV */
#define SCORE_DEPTH_LOSS          4     /* change bit depth */
#define SCORE_ALPHA_LOSS          8     /* lose the alpha channel */
#define SCORE_CHROMA_W_LOSS      16     /* vertical sub-sample */
#define SCORE_CHROMA_H_LOSS      32     /* horizontal sub-sample */
#define SCORE_PALETTE_LOSS       64     /* convert to palette format */
#define SCORE_COLOR_LOSS        128     /* convert to GRAY */

#define COLORSPACE_MASK (GST_VIDEO_FORMAT_FLAG_YUV | \
                         GST_VIDEO_FORMAT_FLAG_RGB | GST_VIDEO_FORMAT_FLAG_GRAY)
#define ALPHA_MASK      (GST_VIDEO_FORMAT_FLAG_ALPHA)
#define PALETTE_MASK    (GST_VIDEO_FORMAT_FLAG_PALETTE)

  gint loss = G_MAXINT32;
  GstVideoFormatFlags in_flags, out_flags;
  const GstVideoFormatInfo *in_info = gst_video_format_get_info(in_name);
  const GstVideoFormatInfo *out_info = gst_video_format_get_info(out_name);

  if (!in_info || !out_info)
    return G_MAXINT32;

  /* Only OpenCL on DPU platform can convert NV12_10LE to NV12) */
  if (HAS_DPU ()) {
    if (in_name == GST_VIDEO_FORMAT_NV12_10LE
        && out_name == GST_VIDEO_FORMAT_NV12)
      return 0;
    else if (in_name != GST_VIDEO_FORMAT_NV12_10LE
        && out_name == GST_VIDEO_FORMAT_NV12)
      return G_MAXINT32;
  }

  /* accept input format immediately without loss */
  if (in_info == out_info) {
    GST_LOG("same format %s", GST_VIDEO_FORMAT_INFO_NAME(in_info));
    return 0;
  }

  loss = SCORE_FORMAT_CHANGE;

  in_flags = GST_VIDEO_FORMAT_INFO_FLAGS (in_info);
  (int&)in_flags &= ~GST_VIDEO_FORMAT_FLAG_LE;
  (int&)in_flags &= ~GST_VIDEO_FORMAT_FLAG_COMPLEX;
  (int&)in_flags &= ~GST_VIDEO_FORMAT_FLAG_UNPACK;

  out_flags = GST_VIDEO_FORMAT_INFO_FLAGS (out_info);
  (int&)out_flags &= ~GST_VIDEO_FORMAT_FLAG_LE;
  (int&)out_flags &= ~GST_VIDEO_FORMAT_FLAG_COMPLEX;
  (int&)out_flags &= ~GST_VIDEO_FORMAT_FLAG_UNPACK;

  if ((out_flags & PALETTE_MASK) != (in_flags & PALETTE_MASK)) {
    loss += SCORE_PALETTE_CHANGE;
    if (out_flags & PALETTE_MASK)
      loss += SCORE_PALETTE_LOSS;
  }

  if ((out_flags & COLORSPACE_MASK) != (in_flags & COLORSPACE_MASK)) {
    loss += SCORE_COLORSPACE_LOSS;
    if (out_flags & GST_VIDEO_FORMAT_FLAG_GRAY)
      loss += SCORE_COLOR_LOSS;
  }

  if ((out_flags & ALPHA_MASK) != (in_flags & ALPHA_MASK)) {
    loss += SCORE_ALPHA_CHANGE;
    if (in_flags & ALPHA_MASK)
      loss += SCORE_ALPHA_LOSS;
  }

  if ((in_info->h_sub[1]) != (out_info->h_sub[1])) {
    loss += SCORE_CHROMA_H_CHANGE;
    if ((in_info->h_sub[1]) < (out_info->h_sub[1]))
      loss += SCORE_CHROMA_H_LOSS;
  }
  if ((in_info->w_sub[1]) != (out_info->w_sub[1])) {
    loss += SCORE_CHROMA_W_CHANGE;
    if ((in_info->w_sub[1]) < (out_info->w_sub[1]))
      loss += SCORE_CHROMA_W_LOSS;
  }

  if ((in_info->bits) != (out_info->bits)) {
    loss += SCORE_DEPTH_CHANGE;
    if ((in_info->bits) > (out_info->bits))
      loss += SCORE_DEPTH_LOSS;
  }

  GST_LOG("%s -> %s, loss = %d", GST_VIDEO_FORMAT_INFO_NAME(in_info),
                  GST_VIDEO_FORMAT_INFO_NAME(out_info), loss);
  return loss;
}

static GstCaps*
caps_from_fmt_list(
  GList* list)
{
  gint i;
  GstCaps *caps = NULL;

  for (i=0; i<g_list_length (list); i++) {
    GstVideoFormat fmt = (GstVideoFormat)(long)g_list_nth_data(list, i);

    if (caps) {
      GstCaps *newcaps = gst_caps_new_simple("video/x-raw",
          "format", G_TYPE_STRING, gst_video_format_to_string(fmt), NULL);
      gst_caps_append (caps, newcaps);
    } else {
      caps = gst_caps_new_simple("video/x-raw",
          "format", G_TYPE_STRING, gst_video_format_to_string(fmt), NULL);
    }
  }

  caps = gst_caps_simplify(caps);
  return caps;
}

static guint
fixate_format_caps(
  GstBaseTransform *transform,
  GstCaps *caps,
  GstCaps *othercaps)
{
  GstStructure *outs;
  GstStructure *tests;
  const GValue *format;
  GstVideoFormat out_fmt = GST_VIDEO_FORMAT_UNKNOWN;
  const GstVideoFormatInfo *out_info = NULL;
  const gchar *fmt_name;
  GstStructure *ins;
  const gchar *in_interlace;
  gboolean interlace = FALSE;
  GstCaps *new_caps;

  GstNnInferenceDemo *demo = (GstNnInferenceDemo *)(transform);
  Imx2DDevice *device = demo->device;

  //the input caps should fixed alreay, and only have caps0
  ins = gst_caps_get_structure(caps, 0);
  outs = gst_caps_get_structure(othercaps, 0);

  in_interlace = gst_structure_get_string(ins, "interlace-mode");
  if (in_interlace && (g_strcmp0(in_interlace, "interleaved") == 0
                       || g_strcmp0(in_interlace, "mixed") == 0)) {
    interlace = TRUE;
  }

  new_caps = gst_caps_copy(othercaps);

  GstVideoFormat in_fmt;
  gint min_loss = G_MAXINT32;
  gint loss;
  guint i, j;

  fmt_name = gst_structure_get_string(ins, "format");
  if (!fmt_name) {
    gst_caps_unref(new_caps);
    return -1;
  }

  GST_LOG("source format : %s", fmt_name);

  in_fmt = gst_video_format_from_string(fmt_name);

  for (i = 0; i < gst_caps_get_size(new_caps); i++) {
    tests = gst_caps_get_structure(new_caps, i);
    format = gst_structure_get_value(tests, "format");
    if (!format) {
      gst_caps_unref(new_caps);
      return -1;
    }

    if (GST_VALUE_HOLDS_LIST(format)) {
      for (j = 0; j < gst_value_list_get_size(format); j++) {
        const GValue *val = gst_value_list_get_value(format, j);
        if (G_VALUE_HOLDS_STRING(val)) {
          out_fmt = gst_video_format_from_string(g_value_get_string(val));
          loss = get_format_conversion_loss(transform, in_fmt, out_fmt);
          if (loss < min_loss) {
            out_info = gst_video_format_get_info(out_fmt);
            min_loss = loss;
          }

          if (min_loss == 0)
            break;
        }
      }
    } else if (G_VALUE_HOLDS_STRING(format)) {
      out_fmt = gst_video_format_from_string(g_value_get_string(format));
      loss = get_format_conversion_loss(transform, in_fmt, out_fmt);
      if (loss < min_loss) {
        out_info = gst_video_format_get_info(out_fmt);
        min_loss = loss;
      }
    }

    if (min_loss == 0)
      break;
  }

  gst_caps_unref(new_caps);

  if (out_info) {
    fmt_name = GST_VIDEO_FORMAT_INFO_NAME(out_info);
    gst_structure_set(outs, "format", G_TYPE_STRING, fmt_name, NULL);
    GST_LOG("out format %s", fmt_name);
    return 0;
  } else {
    gst_structure_set(outs, "format", G_TYPE_STRING, "UNKNOWN", NULL);
    GST_LOG("out format not match");
    return -1;
  }
}

static GstCaps*
fixate_caps(
  GstBaseTransform *transform,
  GstPadDirection direction,
  GstCaps *caps,
  GstCaps *othercaps)
{
  GstStructure *ins, *outs;
  GValue const *from_par, *to_par;
  GValue fpar = { 0, }, tpar = { 0, };
  const gchar *in_format;
  const GstVideoFormatInfo *in_info, *out_info = NULL;
  gint min_loss = G_MAXINT32;
  guint i, capslen;

  g_return_val_if_fail(gst_caps_is_fixed (caps), othercaps);

  othercaps = gst_caps_make_writable(othercaps);

  GST_DEBUG("fixate othercaps: %" GST_PTR_FORMAT, othercaps);
  GST_DEBUG("based on caps: %" GST_PTR_FORMAT, caps);
  GST_DEBUG("direction: %d", direction);

  ins = gst_caps_get_structure(caps, 0);
  outs = gst_caps_get_structure(othercaps, 0);

  from_par = gst_structure_get_value(ins, "pixel-aspect-ratio");
  to_par = gst_structure_get_value(outs, "pixel-aspect-ratio");

  /* If no par info, then set some assuming value  */
  if (!from_par || !to_par) {
    if (direction == GST_PAD_SINK) {
      if (!from_par) {
        g_value_init(&fpar, GST_TYPE_FRACTION);
        gst_value_set_fraction(&fpar, 1, 1);
        from_par = &fpar;
      }
      if (!to_par) {
        g_value_init(&tpar, GST_TYPE_FRACTION_RANGE);
        gst_value_set_fraction_range_full(&tpar, 1, G_MAXINT32, G_MAXINT32, 1);
        to_par = &tpar;
      }
    } else {
      if (!to_par) {
        g_value_init(&tpar, GST_TYPE_FRACTION);
        gst_value_set_fraction(&tpar, 1, 1);
        to_par = &tpar;
        gst_structure_set(outs, "pixel-aspect-ratio",
                          GST_TYPE_FRACTION, 1, 1, NULL);
      }
      if (!from_par) {
        g_value_init(&fpar, GST_TYPE_FRACTION);
        gst_value_set_fraction (&fpar, 1, 1);
        from_par = &fpar;
      }
    }
  }

  /* from_par should be fixed now */
  gint from_w, from_h, from_par_n, from_par_d, to_par_n, to_par_d;
  gint w = 0, h = 0;
  gint from_dar_n, from_dar_d;
  gint num, den;
  GstStructure *tmp;
  gint set_w, set_h, set_par_n, set_par_d;

  from_par_n = gst_value_get_fraction_numerator(from_par);
  from_par_d = gst_value_get_fraction_denominator(from_par);

  gst_structure_get_int(ins, "width", &from_w);
  gst_structure_get_int(ins, "height", &from_h);

  gst_structure_get_int(outs, "width", &w);
  gst_structure_get_int(outs, "height", &h);

  /* if both width and height are already fixed, we can do nothing */
  if (w && h) {
    guint dar_n, dar_d;
    GST_DEBUG("dimensions already set to %dx%d", w, h);

    if (!gst_value_is_fixed(to_par)) {
      if (gst_video_calculate_display_ratio(&dar_n, &dar_d,
          from_w, from_h, from_par_n, from_par_d, w, h)) {
        GST_DEBUG("fixating to_par to %d/%d", dar_n, dar_d);

        if (gst_structure_has_field(outs, "pixel-aspect-ratio")) {
          gst_structure_fixate_field_nearest_fraction(outs,
                                        "pixel-aspect-ratio", dar_n, dar_d);
        } else if (dar_n != dar_d) {
          gst_structure_set(outs, "pixel-aspect-ratio",
                            GST_TYPE_FRACTION, dar_n, dar_d, NULL);
        }
      }
    }

    goto done;
  }

  /* Calculate input DAR */
  gst_util_fraction_multiply(from_w, from_h, from_par_n, from_par_d,
                              &from_dar_n, &from_dar_d);
  GST_LOG("Input DAR is %d/%d", from_dar_n, from_dar_d);

  /* If either width or height are fixed, choose a height or width and PAR */
  if (h) {
    GST_DEBUG("height is fixed (%d)", h);

    /* If the PAR is fixed, choose the width that match DAR */
    if (gst_value_is_fixed(to_par)) {
      to_par_n = gst_value_get_fraction_numerator(to_par);
      to_par_d = gst_value_get_fraction_denominator(to_par);
      GST_DEBUG("PAR is fixed %d/%d", to_par_n, to_par_d);

      gst_util_fraction_multiply(from_dar_n, from_dar_d,
                                 to_par_d, to_par_n, &num, &den);
      w = (guint) gst_util_uint64_scale_int(h, num, den);
      gst_structure_fixate_field_nearest_int(outs, "width", w);
    } else {
      /* The PAR is not fixed, Check if we can keep the input width */
      tmp = gst_structure_copy(outs);
      gst_structure_fixate_field_nearest_int(tmp, "width", from_w);
      gst_structure_get_int(tmp, "width", &set_w);
      gst_util_fraction_multiply(from_dar_n, from_dar_d, h, set_w,
                                 &to_par_n, &to_par_d);

      if (!gst_structure_has_field(tmp, "pixel-aspect-ratio"))
        gst_structure_set_value(tmp, "pixel-aspect-ratio", to_par);

      gst_structure_fixate_field_nearest_fraction(tmp, "pixel-aspect-ratio",
                                                  to_par_n, to_par_d);
      gst_structure_get_fraction(tmp, "pixel-aspect-ratio",
                                  &set_par_n, &set_par_d);
      gst_structure_free(tmp);

      /* Check if the adjusted PAR is accepted */
      if (set_par_n == to_par_n && set_par_d == to_par_d) {
        if (gst_structure_has_field(outs, "pixel-aspect-ratio")
            || set_par_n != set_par_d) {
          gst_structure_set(outs, "width", G_TYPE_INT, set_w,
           "pixel-aspect-ratio", GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
        }
      } else {
        /* scale the width to the new PAR and check if the adjusted width is
         * accepted. If all that fails we can't keep the DAR */
        gst_util_fraction_multiply(from_dar_n, from_dar_d, set_par_d, set_par_n,
                                  &num, &den);

        w = (guint) gst_util_uint64_scale_int(h, num, den);
        gst_structure_fixate_field_nearest_int(outs, "width", w);
        if (gst_structure_has_field(outs, "pixel-aspect-ratio")
            || set_par_n != set_par_d) {
          gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
                            set_par_n, set_par_d, NULL);
        }
      }
    }
  } else if (w) {
    GST_DEBUG("width is fixed (%d)", w);

    /* If the PAR is fixed, choose the height that match the DAR */
    if (gst_value_is_fixed(to_par)) {
      to_par_n = gst_value_get_fraction_numerator(to_par);
      to_par_d = gst_value_get_fraction_denominator(to_par);
      GST_DEBUG("PAR is fixed %d/%d", to_par_n, to_par_d);

      gst_util_fraction_multiply(from_dar_n, from_dar_d, to_par_d, to_par_n,
                                 &num, &den);
      h = (guint) gst_util_uint64_scale_int(w, den, num);
      gst_structure_fixate_field_nearest_int(outs, "height", h);
    } else {
      /* Check if we can keep the input height */
      tmp = gst_structure_copy(outs);
      gst_structure_fixate_field_nearest_int(tmp, "height", from_h);
      gst_structure_get_int(tmp, "height", &set_h);
      gst_util_fraction_multiply(from_dar_n, from_dar_d, set_h, w,
                                 &to_par_n, &to_par_d);

      if (!gst_structure_has_field(tmp, "pixel-aspect-ratio"))
        gst_structure_set_value(tmp, "pixel-aspect-ratio", to_par);
      gst_structure_fixate_field_nearest_fraction(tmp, "pixel-aspect-ratio",
                                                  to_par_n, to_par_d);
      gst_structure_get_fraction(tmp, "pixel-aspect-ratio",
                                 &set_par_n, &set_par_d);
      gst_structure_free(tmp);

      /* Check if the adjusted PAR is accepted */
      if (set_par_n == to_par_n && set_par_d == to_par_d) {
        if (gst_structure_has_field(outs, "pixel-aspect-ratio")
            || set_par_n != set_par_d) {
          gst_structure_set(outs, "height", G_TYPE_INT, set_h,
           "pixel-aspect-ratio", GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
        }
      } else {
        /* scale the height to the new PAR and check if the adjusted width
         * is accepted. If all that fails we can't keep the DAR */
        gst_util_fraction_multiply(from_dar_n, from_dar_d, set_par_d, set_par_n,
                                    &num, &den);

        h = (guint) gst_util_uint64_scale_int(w, den, num);
        gst_structure_fixate_field_nearest_int(outs, "height", h);
        if (gst_structure_has_field(outs, "pixel-aspect-ratio")
            || set_par_n != set_par_d) {
          gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
              set_par_n, set_par_d, NULL);
        }
      }
    }
  } else {
    /* both h and w not fixed */
    if (gst_value_is_fixed(to_par)) {
      gint f_h, f_w;
      to_par_n = gst_value_get_fraction_numerator(to_par);
      to_par_d = gst_value_get_fraction_denominator(to_par);

      /* Calculate scale factor for the PAR change */
      gst_util_fraction_multiply(from_dar_n, from_dar_d, to_par_n, to_par_d,
                                 &num, &den);

      /* Try to keep the input height (because of interlacing) */
      tmp = gst_structure_copy(outs);
      gst_structure_fixate_field_nearest_int(tmp, "height", from_h);
      gst_structure_get_int(tmp, "height", &set_h);
      w = (guint) gst_util_uint64_scale_int(set_h, num, den);
      gst_structure_fixate_field_nearest_int(tmp, "width", w);
      gst_structure_get_int(tmp, "width", &set_w);
      gst_structure_free(tmp);

      if (set_w == w) {
        gst_structure_set(outs, "width", G_TYPE_INT, set_w,
                          "height", G_TYPE_INT, set_h, NULL);
      } else {
        f_h = set_h;
        f_w = set_w;

        /* If the former failed, try to keep the input width at least */
        tmp = gst_structure_copy(outs);
        gst_structure_fixate_field_nearest_int(tmp, "width", from_w);
        gst_structure_get_int(tmp, "width", &set_w);
        h = (guint) gst_util_uint64_scale_int(set_w, den, num);
        gst_structure_fixate_field_nearest_int(tmp, "height", h);
        gst_structure_get_int(tmp, "height", &set_h);
        gst_structure_free(tmp);

        if (set_h == h)
          gst_structure_set(outs, "width", G_TYPE_INT, set_w,
                            "height", G_TYPE_INT, set_h, NULL);
        else
          gst_structure_set(outs, "width", G_TYPE_INT, f_w,
                            "height", G_TYPE_INT, f_h, NULL);
      }
    } else {
      gint tmp2;
      /* width, height and PAR are not fixed but passthrough is not possible */
      /* try to keep the height and width as good as possible and scale PAR */
      tmp = gst_structure_copy(outs);
      gst_structure_fixate_field_nearest_int(tmp, "height", from_h);
      gst_structure_get_int(tmp, "height", &set_h);
      gst_structure_fixate_field_nearest_int(tmp, "width", from_w);
      gst_structure_get_int(tmp, "width", &set_w);

      gst_util_fraction_multiply(from_dar_n, from_dar_d, set_h, set_w,
                                 &to_par_n, &to_par_d);

      if (!gst_structure_has_field(tmp, "pixel-aspect-ratio"))
        gst_structure_set_value(tmp, "pixel-aspect-ratio", to_par);
      gst_structure_fixate_field_nearest_fraction(tmp, "pixel-aspect-ratio",
                                                  to_par_n, to_par_d);
      gst_structure_get_fraction(tmp, "pixel-aspect-ratio",
                                 &set_par_n, &set_par_d);
      gst_structure_free(tmp);

      if (set_par_n == to_par_n && set_par_d == to_par_d) {
        gst_structure_set(outs, "width", G_TYPE_INT, set_w,
                                "height", G_TYPE_INT, set_h, NULL);
        if (gst_structure_has_field(outs, "pixel-aspect-ratio")
            || set_par_n != set_par_d) {
          gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
                            set_par_n, set_par_d, NULL);
        }
      } else {
        /* Otherwise try to scale width to keep the DAR with the set
         * PAR and height */
        gst_util_fraction_multiply(from_dar_n, from_dar_d, set_par_d, set_par_n,
                                   &num, &den);

        w = (guint) gst_util_uint64_scale_int(set_h, num, den);
        tmp = gst_structure_copy(outs);
        gst_structure_fixate_field_nearest_int(tmp, "width", w);
        gst_structure_get_int(tmp, "width", &tmp2);
        gst_structure_free(tmp);

        if (tmp2 == w) {
          gst_structure_set(outs, "width", G_TYPE_INT, tmp2,
                                  "height", G_TYPE_INT, set_h, NULL);
          if (gst_structure_has_field(outs, "pixel-aspect-ratio")
              || set_par_n != set_par_d) {
            gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
                              set_par_n, set_par_d, NULL);
          }
        } else {
          /* then try the same with the height */
          h = (guint) gst_util_uint64_scale_int(set_w, den, num);
          tmp = gst_structure_copy(outs);
          gst_structure_fixate_field_nearest_int(tmp, "height", h);
          gst_structure_get_int(tmp, "height", &tmp2);
          gst_structure_free(tmp);

          if (tmp2 == h) {
            gst_structure_set(outs, "width", G_TYPE_INT, set_w,
                                    "height", G_TYPE_INT, tmp2, NULL);
            if (gst_structure_has_field(outs, "pixel-aspect-ratio")
                || set_par_n != set_par_d) {
              gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
                                set_par_n, set_par_d, NULL);
            }
          } else {
            /* Don't keep the DAR, take the nearest values from the first try */
            gst_structure_set(outs, "width", G_TYPE_INT, set_w,
                                    "height", G_TYPE_INT, set_h, NULL);
            if (gst_structure_has_field(outs, "pixel-aspect-ratio")
                || set_par_n != set_par_d) {
              gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
                                set_par_n, set_par_d, NULL);
            }
          }
        }
      }
    }
  }

done:
  if (from_par == &fpar)
    g_value_unset(&fpar);
  if (to_par == &tpar)
    g_value_unset(&tpar);

  fixate_format_caps(transform, caps, othercaps);
  othercaps = gst_caps_fixate (othercaps);

  GST_DEBUG("fixated othercaps to %" GST_PTR_FORMAT, othercaps);

  return othercaps;
}

static gboolean
filter_meta (
  GstBaseTransform * trans,
  GstQuery * query,
  GType api,
  const GstStructure * params)
{
  /* propose all metadata upstream */
  return TRUE;
}

static void
set_pool_alignment (
  GstCaps *caps,
  GstBufferPool *pool)
{
  GstVideoInfo info;
  GstVideoAlignment alignment;
  GstStructure *config = gst_buffer_pool_get_config(pool);
  gst_video_info_from_caps (&info, caps);

  memset (&alignment, 0, sizeof (GstVideoAlignment));

  gint w = GST_VIDEO_INFO_WIDTH (&info);
  gint h = GST_VIDEO_INFO_HEIGHT (&info);
  if (!ISALIGNED (w, ALIGNMENT) || !ISALIGNED (h, ALIGNMENT)) {
    alignment.padding_right = ALIGNTO (w, ALIGNMENT) - w;
    alignment.padding_bottom = ALIGNTO (h, ALIGNMENT) - h;
  }

  GST_DEBUG ("pool(%p), [%d, %d]:padding_right (%d), padding_bottom (%d)",
      pool, w, h, alignment.padding_right, alignment.padding_bottom);

  if (!gst_buffer_pool_config_has_option (config, \
        GST_BUFFER_POOL_OPTION_VIDEO_META)) {
    gst_buffer_pool_config_add_option (config,
        GST_BUFFER_POOL_OPTION_VIDEO_META);
  }
  if (!gst_buffer_pool_config_has_option (config,
            GST_BUFFER_POOL_OPTION_VIDEO_ALIGNMENT)) {
    gst_buffer_pool_config_add_option (config,
        GST_BUFFER_POOL_OPTION_VIDEO_ALIGNMENT);
  }

  gst_buffer_pool_config_set_video_alignment (config, &alignment);
  gst_buffer_pool_set_config(pool, config);
}

static gboolean
buffer_pool_is_ok (
  GstBufferPool * pool,
  GstCaps * newcaps,
  gint size)
{
  GstCaps *oldcaps;
  GstStructure *config;
  guint bsize;
  gboolean ret;

  config = gst_buffer_pool_get_config (pool);
  gst_buffer_pool_config_get_params (config, &oldcaps, &bsize, NULL, NULL);
  ret = (size <= bsize) && gst_caps_is_equal (newcaps, oldcaps);
  gst_structure_free (config);

  return ret;
}

static GstBufferPool*
create_bufferpool(
  GstNnInferenceDemo *demo,
  GstCaps *caps,
  guint size,
  guint min,
  guint max)
{
  GstBufferPool *pool;
  GstStructure *config;

  pool = gst_video_buffer_pool_new ();
  if (pool) {

    if (!demo->allocator)
      demo->allocator =
          gst_imx_2d_device_allocator_new((gpointer)(demo->device));

    if (!demo->allocator) {
      GST_ERROR ("new imx video convert allocator failed.");
      gst_buffer_pool_set_active (pool, FALSE);
      gst_object_unref (pool);
      return NULL;
    }

    config = gst_buffer_pool_get_config(pool);
    gst_buffer_pool_config_set_params(config, caps, size, min, max);
    gst_buffer_pool_config_set_allocator(config, demo->allocator, NULL);
    gst_buffer_pool_config_add_option(config,
                                      GST_BUFFER_POOL_OPTION_VIDEO_META);
    if (!gst_buffer_pool_set_config(pool, config)) {
      GST_ERROR ("set buffer pool config failed.");
      gst_buffer_pool_set_active (pool, FALSE);
      gst_object_unref (pool);
      return NULL;
    }
  }

  set_pool_alignment(caps, pool);

  GST_LOG ("created a buffer pool (%p).", pool);
  return pool;
}

static gboolean
propose_allocation (
  GstBaseTransform *transform,
  GstQuery *decide_query,
  GstQuery *query)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *)(transform);
  GstBufferPool *pool;
  GstVideoInfo info;
  guint size = 0;
  GstCaps *caps;
  gboolean need_pool;

  /* passthrough, we're done */
  if (decide_query == NULL) {
    GST_DEBUG ("doing passthrough query");
    return gst_pad_peer_query (transform->srcpad, query);
  } else {
    guint i, n_metas;
    /* non-passthrough, copy all metadata, decide_query does not contain the
     * metadata anymore that depends on the buffer memory */
    n_metas = gst_query_get_n_allocation_metas (decide_query);
    for (i = 0; i < n_metas; i++) {
      GType api;
      const GstStructure *params;
      api = gst_query_parse_nth_allocation_meta (decide_query, i, &params);
      gst_query_add_allocation_meta (query, api, params);
    }
  }

  gst_query_parse_allocation (query, &caps, &need_pool);

  if (need_pool) {
    if (caps == NULL) {
      GST_ERROR_OBJECT (demo, "no caps specified.");
      return FALSE;
    }

    if (!gst_video_info_from_caps (&info, caps))
      return FALSE;

    size = GST_VIDEO_INFO_SIZE (&info);
    UNREF_BUFFER (demo->in_buf);
    UNREF_POOL(demo->in_pool);
    GST_DEBUG_OBJECT(demo, "creating new input pool");
    pool = create_bufferpool(demo, caps, size, 1, IN_POOL_MAX_BUFFERS);
    demo->in_pool = pool;
    demo->pool_config_update = TRUE;

    if (pool) {
      GST_DEBUG_OBJECT (demo, "propose_allocation, pool(%p).", pool);
      GstStructure *config = gst_buffer_pool_get_config (pool);
      gst_buffer_pool_config_get_params (config, &caps, &size, NULL, NULL);
      gst_structure_free (config);

      gst_query_add_allocation_pool (query, pool, size, 1, IN_POOL_MAX_BUFFERS);
      gst_query_add_allocation_param (query, demo->allocator, NULL);
    } else {
      return FALSE;
    }
  }

  gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);
  gst_query_add_allocation_meta (query, GST_VIDEO_CROP_META_API_TYPE, NULL);

  return TRUE;
}

static gboolean
decide_allocation (
  GstBaseTransform *transform,
  GstQuery *query)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *)(transform);
  GstCaps *outcaps;
  GstBufferPool *pool = NULL;
  guint size, num, min = 0, max = 0;
  GstStructure *config = NULL;
  GstVideoInfo vinfo;
  gboolean new_pool = TRUE;
  GstAllocator *allocator = NULL;

  gst_query_parse_allocation(query, &outcaps, NULL);
  gst_video_info_init(&vinfo);
  gst_video_info_from_caps(&vinfo, outcaps);
  num = gst_query_get_n_allocation_pools(query);
  size = vinfo.size;

  GST_DEBUG_OBJECT(demo, "number of allocation pools: %d", num);

  /* if downstream element provided buffer pool with phy buffers */
  if (num > 0) {
    guint i = 0;
    for (; i < num; ++i) {
      gst_query_parse_nth_allocation_pool(query, i, &pool, &size, &min, &max);
      if (pool) {
        config = gst_buffer_pool_get_config(pool);
        gst_buffer_pool_config_get_allocator(config, &allocator, NULL);
        if (allocator && GST_IS_ALLOCATOR_PHYMEM(allocator)) {
          size = MAX(size, vinfo.size);
          new_pool = FALSE;
          break;
        } else {
          GST_LOG_OBJECT (demo, "no phy allocator in output pool (%p)", pool);
        }

        if (config) {
          gst_structure_free (config);
          config = NULL;
        }

        allocator = NULL;
        gst_object_unref (pool);
      }
    }
  }

  size = MAX(size, vinfo.size);
  size = PAGE_ALIGN(size);

  if (max == 0)
    if (min < 3)
      max = min = 3;
    else
      max = min;

  /* downstream doesn't provide a pool or the pool has no ability to allocate
   * physical memory buffers, we need create new pool */
  if (new_pool) {
    UNREF_POOL(demo->self_out_pool);
    GST_DEBUG_OBJECT(demo, "creating new output pool");
    pool = create_bufferpool(demo, outcaps, size,
                                                   min, max);
    demo->self_out_pool = pool;
    config = gst_buffer_pool_get_config (pool);
    gst_buffer_pool_set_active(pool, TRUE);
  } else {
    // check the requirement of output alignment
    set_pool_alignment(outcaps, pool);
  }

  demo->out_pool = pool;
  gst_buffer_pool_config_get_params (config, &outcaps, &size, &min, &max);

  GST_DEBUG_OBJECT(demo, "pool config:  outcaps: %" GST_PTR_FORMAT "  "
      "size: %u  min buffers: %u  max buffers: %u", outcaps, size, min, max);
  gst_structure_free (config);

  if (pool) {
    if (num > 0)
      gst_query_set_nth_allocation_pool(query, 0, pool, size, min, max);
    else
      gst_query_add_allocation_pool(query, pool, size, min, max);

    if (!new_pool)
      gst_object_unref (pool);
  }

  return TRUE;
}

static gboolean
set_info (
  GstVideoFilter *filter,
  GstCaps *in,
  GstVideoInfo *in_info,
  GstCaps *out,
  GstVideoInfo *out_info)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *)(filter);
  Imx2DDevice *device = demo->device;
  GstStructure *ins, *outs;
  const gchar *from_interlace;

  if (!device)
    return FALSE;

  ins = gst_caps_get_structure(in, 0);
  outs = gst_caps_get_structure(out, 0);

  /* if interlaced and we enabled deinterlacing, make it progressive */
  from_interlace = gst_structure_get_string(ins, "interlace-mode");
  if (from_interlace &&
      (g_strcmp0(from_interlace, "interleaved") == 0
          || g_strcmp0(from_interlace, "mixed") == 0)) {
  }

  if (IMX_2D_ROTATION_0 != demo->rotate)
    gst_base_transform_set_passthrough((GstBaseTransform*)filter, FALSE);

  GST_DEBUG ("set info from %" GST_PTR_FORMAT " to %" GST_PTR_FORMAT, in, out);

  if (nninferencedemo_init(demo) != 0) {
    GST_ERROR ("Could not initialize NN Inference demo.");
    return FALSE;
  }

  return TRUE;
}

static guint8 *
_get_cached_phyaddr (
  GstMemory * mem)
{
    return (guint8*)gst_mini_object_get_qdata (GST_MINI_OBJECT (mem),
              (GQuark)g_quark_from_static_string ("phyaddr"));
}

static void
_set_cached_phyaddr (
  GstMemory * mem, guint8 * phyadd)
{
  return gst_mini_object_set_qdata (GST_MINI_OBJECT (mem),
                g_quark_from_static_string ("phyaddr"), phyadd, NULL);
}

static GstFlowReturn
transform_frame(
  GstVideoFilter *filter,
  GstVideoFrame *in,
  GstVideoFrame *out)
{
  GstNnInferenceDemo *demo = (GstNnInferenceDemo *)(filter);
  Imx2DDevice *device = demo->device;
  GstVideoFrame *input_frame = in;
  GstPhyMemMeta *phymemmeta = NULL;
  GstCaps *caps;
  GstVideoFrame temp_in_frame;
  Imx2DFrame src = {0}, dst = {0};
  PhyMemBlock src_mem = {0}, dst_mem = {0};
  guint i, n_mem;
  GstVideoCropMeta *in_crop = NULL, *out_crop = NULL;
  GstVideoInfo info;
  GstDmabufMeta *dmabuf_meta;
  gint64 drm_modifier = 0;

  if (!device)
    return GST_FLOW_ERROR;

  if (!(gst_buffer_is_phymem(out->buffer)
        || gst_is_dmabuf_memory (gst_buffer_peek_memory (out->buffer, 0)))) {
    GST_ERROR ("out buffer is not phy memory or DMA Buf");
    return GST_FLOW_ERROR;
  }

  /* Check if need copy input frame */
  if (!(gst_buffer_is_phymem(in->buffer)
        || gst_is_dmabuf_memory (gst_buffer_peek_memory (in->buffer, 0)))) {
    GST_DEBUG ("copy input frame to physical continues memory");
    caps = gst_video_info_to_caps(&in->info);
    gst_video_info_from_caps(&info, caps); //update the size info

    if (!demo->in_pool ||
        !buffer_pool_is_ok(demo->in_pool, caps,info.size)) {
      UNREF_POOL(demo->in_pool);
      GST_DEBUG_OBJECT(demo, "creating new input pool");
      demo->in_pool = create_bufferpool(demo, caps,
          info.size, 1, IN_POOL_MAX_BUFFERS);
    }

    gst_caps_unref (caps);

    if (demo->in_pool && !demo->in_buf) {
      gst_buffer_pool_set_active(demo->in_pool, TRUE);
      GstFlowReturn ret = gst_buffer_pool_acquire_buffer(demo->in_pool,
                                                  &(demo->in_buf), NULL);
      if (ret != GST_FLOW_OK)
        GST_ERROR("error acquiring input buffer: %s", gst_flow_get_name(ret));
      else
        GST_LOG ("created input buffer (%p)", demo->in_buf);
    }

    if (demo->in_buf) {
      gst_video_frame_map(&temp_in_frame, &info, demo->in_buf, GST_MAP_WRITE);
      gst_video_frame_copy(&temp_in_frame, in);
      input_frame = &temp_in_frame;
      gst_video_frame_unmap(&temp_in_frame);
    } else {
      GST_ERROR ("Can't get input buffer");
      return GST_FLOW_ERROR;
    }
  }

  if (demo->pool_config_update) {
    //alignment check
    memset (&demo->in_video_align, 0, sizeof(GstVideoAlignment));
    phymemmeta = GST_PHY_MEM_META_GET (input_frame->buffer);
    if (phymemmeta) {
      demo->in_video_align.padding_right = phymemmeta->x_padding;
      demo->in_video_align.padding_bottom = phymemmeta->y_padding;
      GST_DEBUG_OBJECT (demo, "physical memory meta x_padding: %d y_padding: %d",
          phymemmeta->x_padding, phymemmeta->y_padding);
    } else if (demo->in_pool) {
      GstStructure *config = gst_buffer_pool_get_config (demo->in_pool);
      memset (&demo->in_video_align, 0, sizeof(GstVideoAlignment));

      if (gst_buffer_pool_config_has_option (config,
          GST_BUFFER_POOL_OPTION_VIDEO_ALIGNMENT)) {
        gst_buffer_pool_config_get_video_alignment (config,
            &demo->in_video_align);
        GST_DEBUG ("input pool has alignment (%d, %d) , (%d, %d)",
          demo->in_video_align.padding_left,
          demo->in_video_align.padding_top,
          demo->in_video_align.padding_right,
          demo->in_video_align.padding_bottom);
      }

      gst_structure_free (config);
    }

    if (demo->out_pool) {
      GstStructure *config = gst_buffer_pool_get_config (demo->out_pool);
      memset (&demo->out_video_align, 0, sizeof(GstVideoAlignment));

      if (gst_buffer_pool_config_has_option (config,
          GST_BUFFER_POOL_OPTION_VIDEO_ALIGNMENT)) {
        gst_buffer_pool_config_get_video_alignment (config,
            &demo->out_video_align);
        GST_DEBUG ("output pool has alignment (%d, %d) , (%d, %d)",
          demo->out_video_align.padding_left,
          demo->out_video_align.padding_top,
          demo->out_video_align.padding_right,
          demo->out_video_align.padding_bottom);
      }

      gst_structure_free (config);
    }

    /* set physical memory padding info */
    if (demo->self_out_pool && gst_buffer_is_writable (out->buffer)) {
      phymemmeta = GST_PHY_MEM_META_ADD (out->buffer);
      phymemmeta->x_padding = demo->out_video_align.padding_right;
      phymemmeta->y_padding = demo->out_video_align.padding_bottom;
      GST_DEBUG_OBJECT (demo, "out physical memory meta x_padding: %d y_padding: %d",
          phymemmeta->x_padding, phymemmeta->y_padding);
    }

    demo->pool_config_update = FALSE;
  }

  caps = gst_pad_get_current_caps (GST_BASE_TRANSFORM_SINK_PAD(demo));
  gst_video_info_from_caps(&info, caps);
  gst_caps_unref (caps);

  src.info.fmt = GST_VIDEO_INFO_FORMAT(&(in->info));
  src.info.w = in->info.width + demo->in_video_align.padding_left +
              demo->in_video_align.padding_right;
  src.info.h = in->info.height + demo->in_video_align.padding_top +
              demo->in_video_align.padding_bottom;
  src.info.stride = in->info.stride[0];

  dmabuf_meta = gst_buffer_get_dmabuf_meta (in->buffer);
  if (dmabuf_meta) {
    drm_modifier = dmabuf_meta->drm_modifier;
    dmabuf_meta->drm_modifier = 0;
  }

  dmabuf_meta = gst_buffer_get_dmabuf_meta (out->buffer);
  if (dmabuf_meta) {
    dmabuf_meta->drm_modifier = 0;
  }

  GST_TRACE ("buffer modifier type %ld", drm_modifier);

  if (drm_modifier == DRM_FORMAT_MOD_AMPHION_TILED)
    src.info.tile_type = IMX_2D_TILE_AMHPION;

  gint ret = device->config_input(device, &src.info);

  GST_LOG ("Input: %s, %dx%d(%d)", GST_VIDEO_FORMAT_INFO_NAME(in->info.finfo),
      src.info.w, src.info.h, src.info.stride);

  dst.info.fmt = GST_VIDEO_INFO_FORMAT(&(out->info));
  dst.info.w = out->info.width + demo->out_video_align.padding_left +
                demo->out_video_align.padding_right;
  dst.info.h = out->info.height + demo->out_video_align.padding_top +
                demo->out_video_align.padding_bottom;
  dst.info.stride = out->info.stride[0];

  ret |= device->config_output(device, &dst.info);

  GST_LOG ("Output: %s, %dx%d", GST_VIDEO_FORMAT_INFO_NAME(out->info.finfo),
      out->info.width, out->info.height);

  if (ret != 0)
    return GST_FLOW_ERROR;

  src.fd[0] = src.fd[1] = src.fd[2] = src.fd[3] = -1;
  if (gst_is_dmabuf_memory (gst_buffer_peek_memory (input_frame->buffer, 0))) {
    src.mem = &src_mem;
    n_mem = gst_buffer_n_memory (input_frame->buffer);
    for (i = 0; i < n_mem; i++)
      src.fd[i] = gst_dmabuf_memory_get_fd (gst_buffer_peek_memory (input_frame->buffer, i));
  } else
    src.mem = gst_buffer_query_phymem_block (input_frame->buffer);
  src.alpha = 0xFF;
  src.crop.x = 0;
  src.crop.y = 0;
  src.crop.w = info.width;
  src.crop.h = info.height;
  src.rotate = demo->rotate;

  in_crop = gst_buffer_get_video_crop_meta(in->buffer);
  if (in_crop != NULL) {
    GST_LOG ("input crop meta: (%d, %d, %d, %d).", in_crop->x, in_crop->y,
        in_crop->width, in_crop->height);
    if ((in_crop->x >= info.width) || (in_crop->y >= info.height))
      return GST_FLOW_ERROR;

    src.crop.x += in_crop->x;
    src.crop.y += in_crop->y;
    src.crop.w = MIN(in_crop->width, (info.width - in_crop->x));
    src.crop.h = MIN(in_crop->height, (info.height - in_crop->y));
  }

  //rotate and de-interlace setting
  if (device->set_rotate(device, demo->rotate) < 0) {
    GST_WARNING_OBJECT (demo, "set rotate failed");
    return GST_FLOW_ERROR;
  }

  if (gst_is_dmabuf_memory (gst_buffer_peek_memory (out->buffer, 0))) {
    dst.mem = &dst_mem;
    n_mem = gst_buffer_n_memory (out->buffer);
    for (i = 0; i < n_mem; i++)
      dst.fd[i] = gst_dmabuf_memory_get_fd (gst_buffer_peek_memory (out->buffer, i));
  } else
    dst.mem = gst_buffer_query_phymem_block (out->buffer);
  dst.alpha = 0xFF;
  dst.interlace_type = IMX_2D_INTERLACE_PROGRESSIVE;
  dst.crop.x = 0;
  dst.crop.y = 0;
  dst.crop.w = out->info.width;
  dst.crop.h = out->info.height;

  out_crop = gst_buffer_get_video_crop_meta(out->buffer);
  if (out_crop != NULL) {
    GST_LOG ("output crop meta: (%d, %d, %d, %d).", out_crop->x, out_crop->y,
        out_crop->width, out_crop->height);
    if ((out_crop->x >= out->info.width) || (out_crop->y >= out->info.height))
      return GST_FLOW_ERROR;

    dst.crop.x += out_crop->x;
    dst.crop.y += out_crop->y;
    dst.crop.w = MIN(out_crop->width, (out->info.width - out_crop->x));
    dst.crop.h = MIN(out_crop->height, (out->info.height - out_crop->y));
  }

  if (!src.mem->paddr)
    src.mem->paddr = _get_cached_phyaddr (gst_buffer_peek_memory (input_frame->buffer, 0));
  if (!src.mem->user_data && src.fd[1] >= 0)
	  src.mem->user_data = (void**)_get_cached_phyaddr (gst_buffer_peek_memory (input_frame->buffer, 1));
  if (!dst.mem->paddr)
    dst.mem->paddr = _get_cached_phyaddr (gst_buffer_peek_memory (out->buffer, 0));

  //convert
  if (device->convert(device, &dst, &src) == 0) {
    GST_TRACE ("frame conversion done");

    if (nninference((GObject*)demo, &info, &src, &dst) != 0) {
      return GST_FLOW_ERROR;
    }

    if (!_get_cached_phyaddr (gst_buffer_peek_memory (input_frame->buffer, 0)))
      _set_cached_phyaddr (gst_buffer_peek_memory (input_frame->buffer, 0), src.mem->paddr);
    if (src.fd[1] >= 0 && !_get_cached_phyaddr ((GstMemory*)gst_buffer_peek_memory (input_frame->buffer, 1)))
      _set_cached_phyaddr (gst_buffer_peek_memory (input_frame->buffer, 1), (guint8*)src.mem->user_data);
    if (!_get_cached_phyaddr (gst_buffer_peek_memory (out->buffer, 0)))
      _set_cached_phyaddr (gst_buffer_peek_memory (out->buffer, 0), (guint8*)dst.mem->paddr);

    return GST_FLOW_OK;
  }

  return GST_FLOW_ERROR;
}

static GstFlowReturn
transform_frame_ip(
  GstVideoFilter *filter,
  GstVideoFrame *in)
{
  // FIXME: this is a temporary code. need to improve.
  return transform_frame(filter, in, in);
}

static void
class_init (
  GstNnInferenceDemoClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);
  GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS(klass);
  GstCaps *caps;

  Imx2DDeviceInfo *in_plugin = (Imx2DDeviceInfo *)
      g_type_get_qdata (G_OBJECT_CLASS_TYPE (klass), PARAMS_QDATA);
  g_assert (in_plugin != NULL);

  Imx2DDevice* dev = in_plugin->create(in_plugin->device_type);
  if (!dev)
    return;

  gchar longname[64] = {0};
  gchar desc[64] = {0};
  gint capabilities = dev->get_capabilities(dev);

  snprintf(longname, 64, "i.MX NN Inference demo (%s)", in_plugin->name);
  snprintf(desc, 64, "i.MX NN Inference demo.");
  gst_element_class_set_static_metadata (element_class, longname, "Filter/Converter/Video", desc, "nxp.com");

  GList *list = dev->get_supported_in_fmts(dev);
  caps = caps_from_fmt_list(list);
  g_list_free(list);

  if (!caps) {
    GST_ERROR ("Couldn't create caps for device '%s'", in_plugin->name);
    caps = gst_caps_new_empty_simple ("video/x-raw");
  }
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));

#ifdef PASSTHOUGH_FOR_UNSUPPORTED_OUTPUT_FORMAT
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                            gst_caps_copy(caps)));
#else
  list = dev->get_supported_out_fmts(dev);
  caps = caps_from_fmt_list(list);
  g_list_free(list);

  if (!caps) {
    GST_ERROR ("Couldn't create caps for device '%s'", in_plugin->name);
    caps = gst_caps_new_empty_simple ("video/x-raw");
  }
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
#endif
  klass->in_plugin = in_plugin;

  parent_class = (GstElementClass *)g_type_class_peek_parent (klass);

  gobject_class->finalize = finalize;
  gobject_class->set_property = set_property;
  gobject_class->get_property = get_property;

  if (capabilities & IMX_2D_DEVICE_CAP_ROTATE) {
    g_object_class_install_property (gobject_class, PROP_OUTPUT_ROTATE,
        g_param_spec_enum("rotation", "Output rotation",
          "Rotation that shall be applied to output frames",
          rotation_get_type(),
          ROTATION_DEFAULT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  }

  g_object_class_install_property (gobject_class, PROP_DEMO_MODE,
      g_param_spec_enum("demo-mode", "NN Inference demo mode",
        "Select from \"posenet\", \"mobilenet-ssd\" or \"benchmark\"",
        demo_mode_get_type(),
        DEMO_MODE_DEFAULT,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DISPLAY_STATS,
      g_param_spec_boolean("display-stats", "Display demo stats",
        "Displaying demo stats",
        DISPLAY_STATS_DEFAULT,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE_INFERENCE,
      g_param_spec_boolean("enable-inference", "Enable inference",
        "Enable inference",
        ENABLE_INFERENCE_DEFAULT,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_USE_NNAPI,
      g_param_spec_int("use-nnapi", "Use NNAPI",
        "Use NNAPI",
        0, 2, USE_NNAPI_DEFAULT,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NUM_THREADS,
      g_param_spec_int("num-threads", "Number of threads for inference",
        "Number of threads for inference",
        1, 32, NUM_THREADS_DEFAULT,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "NN Inference model", "Path of the NN Inference model file",
        MODEL_DEFAULT,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LABEL,
      g_param_spec_string ("label", "NN Inference label", "Path of the NN Inference label file",
        LABEL_DEFAULT,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));


  in_plugin->destroy(dev);

  base_transform_class->src_event =
      GST_DEBUG_FUNCPTR(src_event);
  base_transform_class->transform_caps =
      GST_DEBUG_FUNCPTR(transform_caps);
  base_transform_class->fixate_caps =
      GST_DEBUG_FUNCPTR(fixate_caps);
  base_transform_class->filter_meta =
      GST_DEBUG_FUNCPTR (filter_meta);
  base_transform_class->propose_allocation =
      GST_DEBUG_FUNCPTR(propose_allocation);
  base_transform_class->decide_allocation =
      GST_DEBUG_FUNCPTR(decide_allocation);
  video_filter_class->set_info =
      GST_DEBUG_FUNCPTR(set_info);
  video_filter_class->transform_frame =
      GST_DEBUG_FUNCPTR(transform_frame);
  video_filter_class->transform_frame_ip =
      GST_DEBUG_FUNCPTR(transform_frame_ip);

  base_transform_class->passthrough_on_same_caps = TRUE;
}

static void
init (
  GstNnInferenceDemo * demo)
{
  GstNnInferenceDemoClass *klass =
      (GstNnInferenceDemoClass *) G_OBJECT_GET_CLASS (demo);

  if (klass->in_plugin)
    demo->device = klass->in_plugin->create(klass->in_plugin->device_type);

  if (demo->device) {
    if (demo->device->open(demo->device) < 0) {
      GST_ERROR ("Open video process device failed.");
    } else {
      demo->in_buf = NULL;
      demo->in_pool = NULL;
      demo->out_pool = NULL;
      demo->self_out_pool = NULL;
      demo->pool_config_update = TRUE;
      demo->rotate = ROTATION_DEFAULT;
    }
  } else {
    GST_ERROR ("Create video process device failed.");
  }

  demo->demo_mode = DEMO_MODE_DEFAULT;
  demo->model = NULL;
  demo->label = NULL;
  demo->display_stats = DISPLAY_STATS_DEFAULT;
  demo->use_nnapi = USE_NNAPI_DEFAULT;
  demo->enable_inference = ENABLE_INFERENCE_DEFAULT;
  demo->num_threads = NUM_THREADS_DEFAULT;
}

static gboolean
register_plugin (
  GstPlugin * plugin)
{
  GTypeInfo tinfo = {
    sizeof(GstNnInferenceDemoClass),
    NULL,
    NULL,
    (GClassInitFunc)class_init,
    NULL,
    NULL,
    sizeof(GstNnInferenceDemo),
    0,
    (GInstanceInitFunc)init,
  };

  GType type;
  const gchar *name = "nninferencedemo";

  const Imx2DDeviceInfo *in_plugin = imx_get_2d_devices();

  while (in_plugin->name) {
    GST_LOG ("Registering %s video converter", in_plugin->name);

    if (!in_plugin->is_exist()) {
      GST_WARNING("device %s not exist", in_plugin->name);
      in_plugin++;
      continue;
    }

    type = g_type_from_name (name);
    if (!type) {
      type = g_type_register_static (GST_TYPE_VIDEO_FILTER, name, &tinfo, (GTypeFlags)0);
      g_type_set_qdata (type, PARAMS_QDATA, (gpointer) in_plugin);
    }

    if (!gst_element_register (plugin, name, IMX_GST_PLUGIN_RANK, type)) {
      GST_ERROR ("Failed to register %s", name);
      return FALSE;
    }

    in_plugin++;
  }

  return TRUE;
}

static gboolean
plugin_init (
  GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (
    nninferencedemo_debug, "nninferencedemo", 0, "i.MX NN Inference demo element");
  return register_plugin (plugin);
}

GST_PLUGIN_DEFINE(
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  nninferencedemo,
  "i.MX NN Inference demo plugin",
  plugin_init,
  VERSION,
  IMX_GST_PLUGIN_LICENSE,
  "i.MX NN Inference demo plugin",
  "nxp.com");
