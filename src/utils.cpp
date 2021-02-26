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

#include "utils.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace utils {

void
bgrx_to_rgb_row(
  uint8_t *src,
  uint8_t *dst,
  int num_of_pixels)
{
#ifdef __aarch64__
  int num_of_16pix_loop = num_of_pixels >> 4;
  int num_of_8pix_loop = (num_of_pixels & (16-1)) >> 3;
  int num_of_1pix_loop = (num_of_pixels & (8-1));

  for (int i = 0; i < num_of_16pix_loop; i++)
  {
    // load 16pixel (64bytes)
    uint8x16x4_t v_bgrx = vld4q_u8(src);
    uint8x16x3_t v_rgb;
    v_rgb.val[0] = v_bgrx.val[2];
    v_rgb.val[1] = v_bgrx.val[1];
    v_rgb.val[2] = v_bgrx.val[0];
    // store 16pixel (48bytes)
    vst3q_u8(dst, v_rgb);
    src += 64;
    dst += 48;
  }
  for (int i = 0; i < num_of_8pix_loop; i++)
  {
    // load 8pixel (32bytes)
    uint8x8x4_t v_bgrx = vld4_u8(src);
    uint8x8x3_t v_rgb;
    v_rgb.val[0] = v_bgrx.val[2];
    v_rgb.val[1] = v_bgrx.val[1];
    v_rgb.val[2] = v_bgrx.val[0];
    // store 8pixel (24bytes)
    vst3_u8(dst, v_rgb);
    src += 32;
    dst += 24;
  }
#endif
  for (int i = 0; i < num_of_1pix_loop; i++)
  {
    // copy 1 pixel (load 4bytes, store 3bytes)
    dst[0] = src[2];//R
    dst[1] = src[1];//G
    dst[2] = src[0];//B
    src += 4;
    dst += 3;
  }
}

void
bgrx_to_rgb(
  uint8_t *src,
  uint8_t *dst,
  int width,  // pixel
  int height, // pixel
  int stride) // pixel
{
  for (int row = 0; row < height; row++)
  {
    bgrx_to_rgb_row(src, dst, width);
    src += (stride * 4); // increment 4bytes(BGRX8888) * stride
    dst += (width * 3);
  }
}

}
