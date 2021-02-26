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

if ENABLE_VNN
OVXLIB_CFLAGS = -I$(SDKTARGETSYSROOT)/usr/include/OVXLIB
OVXLIB_LIBS = -lovxlib -lOpenVX -lOpenVXU -lCLC -lVSC -lGAL
endif

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

HEADERS_VNN = \
  vnn_inference.h \
  vnn_global.h \
  vnn_tinyyolov3u8.h \
  vnn_tiny_yolov3.h \
  vnn_unetu8.h \
  vnn_unet.h

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

SOURCES_VNN = \
  vnn_inference.cpp \
  vnn_tinyyolov3u8.c \
  vnn_tiny_yolov3.cpp \
  vnn_unetu8.c \
  vnn_unet.cpp

if ENABLE_VNN
noinst_HEADERS = $(HEADERS_TF) $(HEADERS_VNN)
libgstnninferencedemo_la_SOURCES = $(SOURCES_TF) $(SOURCES_VNN)
else
noinst_HEADERS = $(HEADERS_TF)
libgstnninferencedemo_la_SOURCES = $(SOURCES_TF)
endif

libgstnninferencedemo_la_CFLAGS = \
  -DUSE_G2D \
  $(GST_PLUGINS_BASE_CFLAGS) \
  $(OVXLIB_CFLAGS)

if USE_ION
libgstnninferencedemo_la_CFLAGS += -DUSE_ION
endif

if ENABLE_VNN
libgstnninferencedemo_la_CFLAGS += -DENABLE_VNN
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
  $(OVXLIB_LIBS)

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
  $(top_srcdir)/COPYING

# don't install libgstnninferencedemo.la and vnn sources
install-data-hook:
	@(cd $(DESTDIR)/$(plugindir) && $(RM) $(plugin_LTLIBRARIES))

# don't distribute vnn sources if not defined ENABLE_VNN
if !ENABLE_VNN
dist-hook:
	cd $(distdir) && rm -f $(HEADERS_VNN) $(SOURCES_VNN)
endif
