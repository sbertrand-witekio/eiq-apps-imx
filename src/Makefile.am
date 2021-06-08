# Makefile.am for i.MX GStreamer NN Inference demo plugin
#
# SPDX-License-Identifier: LGPL-2.0+
# Copyright 2021 NXP

# OpenCV
#OPENCV_CXXFLAGS = `pkg-config --cflags opencv4`
OPENCV_CXXFLAGS = \
  -std=c++11 \
  -I$(SDKTARGETSYSROOT)/usr/include/opencv4

OPENCV_LIBS = \
  -lopencv_highgui
# -lopencv_core \
# -lopencv_imgproc \
# -lopencv_videoio \
# -lopencv_imgcodecs

# Tensorflow Lite
TFLITE_CXXFLAGS = \
  -I$(SDKTARGETSYSROOT)/usr/include/tensorflow/lite/tools/make/downloads/absl \
  -I$(SDKTARGETSYSROOT)/usr/include/tensorflow/lite/tools/make/downloads/flatbuffers/include

TFLITE_LIBS = -ltensorflow-lite

##############################################################################
# libgstnninferencedemo.la
##############################################################################
plugin_LTLIBRARIES = libgstnninferencedemo.la

HEADERS_TF = \
  gstnninferencedemo.h \
  inference.h \
  tflite_inference.h \
  tflite_benchmark.h \
  posenet.h \
  mobilenet_ssd.h \
  utils.h \
  \
  gstimx.h \
  imx_2d_device.h \
  imx_2d_device_allocator.h

SOURCES_TF = \
  gstnninferencedemo.cpp \
  inference.cpp \
  tflite_inference.cpp \
  tflite_benchmark.cpp \
  posenet.cpp \
  mobilenet_ssd.cpp \
  utils.cpp \
  \
  gstimxcommon.c \
  imx_2d_device_allocator.c \
  imx_2d_device.c \
  imx_2d_device_g2d.c

noinst_HEADERS = $(HEADERS_TF)
libgstnninferencedemo_la_SOURCES = $(SOURCES_TF)

libgstnninferencedemo_la_CFLAGS = \
  -DUSE_G2D \
  $(GST_PLUGINS_BASE_CFLAGS) \
  $(OVXLIB_CFLAGS)

if USE_ION
libgstnninferencedemo_la_CFLAGS += -DUSE_ION
endif

libgstnninferencedemo_la_CXXFLAGS = \
  $(libgstnninferencedemo_la_CFLAGS) \
  $(OPENCV_CXXFLAGS) \
  $(TFLITE_CXXFLAGS)

libgstnninferencedemo_la_LIBADD = \
  $(GST_PLUGINS_BASE_LIBS) \
  -lgstallocators-$(GST_API_VERSION) \
  -lgstvideo-$(GST_API_VERSION) \
  -lg2d \
  $(TFLITE_LIBS) \
  $(OPENCV_LIBS) \
  $(OVXLIB_LIBS) \
  -lgooglecoraledgetpuposenet

libgstnninferencedemo_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstnninferencedemo_la_LIBTOOLFLAGS = $(GST_PLUGIN_LIBTOOLFLAGS)


# package name
PACKAGE_NAME=gstnninferencedemo

# scripts
dist_bin_SCRIPTS= \
  $(PACKAGE_NAME)-mobilenet-ssd-camera \
  $(PACKAGE_NAME)-mobilenet-ssd-video \
  $(PACKAGE_NAME)-posenet-camera \
  $(PACKAGE_NAME)-posenet-video

# README
pkgdatadir = $(datadir)/$(PACKAGE_NAME)
dist_pkgdata_DATA = \
  $(top_srcdir)/README \
  $(top_srcdir)/COPYING-LGPL2.0 \
  $(top_srcdir)/COPYING-LGPL2.1

# don't install libgstnninferencedemo.la
install-data-hook:
	@(cd $(DESTDIR)/$(plugindir) && $(RM) $(plugin_LTLIBRARIES))
