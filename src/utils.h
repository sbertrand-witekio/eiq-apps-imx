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

#ifndef utils_h
#define utils_h

#include <stdint.h>

namespace utils {

  void bgrx_to_rgb_row(
    uint8_t *src,
    uint8_t *dst,
    int num_of_pixels);

  void bgrx_to_rgb(
    uint8_t *src,
    uint8_t *dst,
    int width,   // pixel
    int height,  // pixel
    int stride); // pixel

}

#endif
