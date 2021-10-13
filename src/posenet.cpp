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

#include "posenet.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

GST_DEBUG_CATEGORY(posenet_t_debug);
#define GST_CAT_DEFAULT posenet_t_debug


static
const char *keypoint_name[POSE_NUM_KEYPOINTS] =
{
  "NOSE",           /*  0 */
  "LEFT_EYE",       /*  1 */
  "RIGHT_EYE",      /*  2 */
  "LEFT_EAR",       /*  3 */
  "RIGHT_EAR",      /*  4 */
  "LEFT_SHOULDER",  /*  5 */
  "RIGHT_SHOULDER", /*  6 */
  "LEFT_ELBOW",     /*  7 */
  "RIGHT_ELBOW",    /*  8 */
  "LEFT_WRIST",     /*  9 */
  "RIGHT_WRIST",    /* 10 */
  "LEFT_HIP",       /* 11 */
  "RIGHT_HIP",      /* 12 */
  "LEFT_KNEE",      /* 13 */
  "RIGHT_KNEE",     /* 14 */
  "LEFT_ANKLE",     /* 15 */
  "RIGHT_ANKLE"     /* 16 */
};

void posenet_t::parse_pose(
  pose_results& results,
  int image_width,
  int image_height,
  int wanted_width,
  int wanted_height)
{
  size_t sz[4] = {0, };
  float *keypoint_coord = (float *)(typed_output_tensor<float>(0, &sz[0]));
  float *keypoint_score = (float *)(typed_output_tensor<float>(1, &sz[1]));
  float *pose_score = (float *)(typed_output_tensor<float>(2, &sz[2]));
  float *npose_f = (float *)(typed_output_tensor<float>(3, &sz[3]));

  GST_TRACE("posenet: %p[%ld], %p[%ld], %p[%ld], %p[%ld]",
        keypoint_coord, sz[0],
        keypoint_score, sz[1],
        pose_score, sz[2],
        npose_f, sz[3]);

  results.n_pose_ = (int)(*npose_f);
  for (int i = 0; i < results.n_pose_; i++) {
    results.pose_[i].score_ = pose_score[i];
    for (int j = 0; j < POSE_NUM_KEYPOINTS; j++) {
      results.pose_[i].pt_[j].y_ = *keypoint_coord++ * image_height / wanted_height;
      results.pose_[i].pt_[j].x_ = *keypoint_coord++ * image_width / wanted_width;
      results.pose_[i].pt_[j].score_ = *keypoint_score++;
    }
  }
}

void posenet_t::draw_keypoint(
  cv::Mat& frame,
  pose_keypoint& point)
{
  cv::Point pt;
  pt.x = point.x_;
  pt.y = point.y_;
  //cv::circle(frame, pt, 4, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
  cv::circle(frame, pt, 4, cv::Scalar(255, 255, 0), 2);
}

void posenet_t::draw_body_line(
  cv::Mat& frame,
  pose_keypoint& start,
  pose_keypoint& end)
{
  cv::Point pt_start;
  pt_start.x = start.x_;
  pt_start.y = start.y_;
  cv::Point pt_end;
  pt_end.x = end.x_;
  pt_end.y = end.y_;
  //cv::line(frame, pt_start, pt_end, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
  cv::line(frame, pt_start, pt_end, cv::Scalar(255, 255, 0), 2);
}

void posenet_t::draw_pose(
  cv::Mat& frame,
  pose_results& results,
  float pose_threshold,
  float keypoint_threshold)
{
  for (int n = 0; n < results.n_pose_; n++) {
    if (results.pose_[n].score_ > pose_threshold) {
      for (int i = 0; i < POSE_NUM_KEYPOINTS; i++) {
        if (results.pose_[n].pt_[i].score_ > keypoint_threshold) {
          draw_keypoint(frame, results.pose_[n].pt_[i]);
        }
      }
      struct Line {
        int start;
        int end;
      } lines[] = {
        { 5,  6}, // Left Shoulder - Right Shoulder
        { 5, 11}, // Left Shoulder - Left Hip
        { 6, 12}, // Right Shoulder - Right Hip
        {11, 12}, // Left Hip - Right Hip
        { 5,  7}, // Left Shoulder - Left Elbow
        { 7,  9}, // Left Elbow - Left Wrist
        { 6,  8}, // Right Shoulder - Right Elbow
        { 8, 10}, // Right Elbow - Right Wrist
        {11, 13}, // Left Hip - Left Knee
        {13, 15}, // Left Knee - Left Ankle
        {12, 14}, // Right Hip - Right Knee
        {14, 16}, // Right Knee - Right Ankle
      };
      for (int i = 0; i < sizeof(lines)/sizeof(struct Line); i++) {
        if ((results.pose_[n].pt_[lines[i].start].score_ > keypoint_threshold) &&
          (results.pose_[n].pt_[lines[i].end].score_ > keypoint_threshold)) {
          draw_body_line(frame, results.pose_[n].pt_[lines[i].start], results.pose_[n].pt_[lines[i].end]);
        }
      }
    }
  }
}


posenet_t::posenet_t()
{
  GST_DEBUG_CATEGORY_INIT(posenet_t_debug, "posenet_t", 0, "i.MX NN Inference demo TFLite posenet class");
  GST_TRACE("%s", __func__);
}

posenet_t::~posenet_t()
{
  GST_TRACE("%s", __func__);
}

int posenet_t::init(
  const std::string& model,
  int use_nnapi,
  int num_threads)
{
  GST_TRACE("%s", __func__);
  return tflite_inference_t::init(model, use_nnapi, num_threads);
}

int posenet_t::draw_results(cv::Mat& frame)
{
  GST_TRACE("%s", __func__);

  pose_results results;
  parse_pose(results, frame.cols, frame.rows, bgrx_width_, bgrx_height_);
  float pose_threshold = 0.3;
  float keypoint_threshold = 0.3;
  draw_pose(frame, results, pose_threshold, keypoint_threshold);

  return OK;
}
