#ifndef TIGR_RESIZE_H
#define TIGR_RESIZE_H

#include "tigr.h"

typedef struct TigrResize {
  Tigr *window;
  Tigr *contents;
  Tigr *contents_display;

  TPixel bg_colour;

  int last_w, last_h;
} TigrResize;

void tigrResizeUpdate(TigrResize *resize);

#endif