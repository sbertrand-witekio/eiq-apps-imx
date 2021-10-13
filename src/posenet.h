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

#ifndef posenet_h
#define posenet_h

#include "tflite_inference.h"

#define POSE_NUM_KEYPOINTS (17)
#define POSE_NUM_POSE_MAX (10)
// pose keypoint
struct pose_keypoint {
  float score_;
  float x_;
  float y_;
};
// pose structure
struct pose_structure {
  float score_;
  pose_keypoint pt_[POSE_NUM_KEYPOINTS];
};
// pose results
struct pose_results {
  int n_pose_;
  pose_structure pose_[POSE_NUM_POSE_MAX];
};

class posenet_t : public tflite_inference_t
{
public:

  enum {
    OK = 0,
    ERROR = -1,
  };

  posenet_t();

  virtual ~posenet_t();

  int init(
    const std::string& model,
    int use_nnapi = 2,
    int num_threads = 4);

  virtual int draw_results(cv::Mat& frame);

private:

  void parse_pose(
    pose_results& results,
    int image_width,
    int image_height,
    int wanted_width,
    int wanted_height);

  void draw_keypoint(
    cv::Mat& frame,
    pose_keypoint& point);

  void draw_body_line(
    cv::Mat& frame,
    pose_keypoint& start,
    pose_keypoint& end);

  void draw_pose(
    cv::Mat& frame,
    pose_results& results,
    float pose_threshold,
    float keypoint_threshold);

  // unused
  posenet_t(const posenet_t&);
  posenet_t& operator=(const posenet_t&);

};

#endif
