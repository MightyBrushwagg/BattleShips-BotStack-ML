#include "tigr_resize.h"

#include <stdio.h>
#include <stdlib.h>

#define MIN(v1, v2) (v1) * (v1 <= v2) + (v2) * (v2 < v1)
#define MAX(v1, v2) (v1) * (v1 >= v2) + (v2) * (v2 > v1)

// Scale a provided Tigr bitmap to the size [dx,dy] using bilinear interpolation
Tigr *scaleBitmap(Tigr *src, int dx, int dy);

static void lockAspectRatio(int *out_w, int *out_h, int w, int h, int src_w, int src_h) {
  double ratio = MIN((double)w / src_w, (double)h / src_h);

  *out_w = src_w * ratio;
  *out_h = src_h * ratio;
}

void tigrResizeUpdate(TigrResize *r) {
  static int w, h;
  if (r->last_w != r->window->w || r->last_h != r->window->h) {
    r->last_w = r->window->w, r->last_h = r->window->h;
    lockAspectRatio(&w, &h, r->last_w, r->last_h, r->contents->w, r->contents->h);
  }

  tigrClear(r->window, r->bg_colour);

  if (r->contents_display != NULL)
    tigrFree(r->contents_display);
  r->contents_display = scaleBitmap(r->contents, w, h);

  tigrBlit(r->window, r->contents_display, MAX((r->window->w - r->contents_display->w) / 2, 0),
           MAX((r->window->h - r->contents_display->h) / 2, 0), 0, 0, r->contents_display->w, r->contents_display->h);
  tigrUpdate(r->window);
}

Tigr *scaleBitmap(Tigr *src, int dw, int dh) {
  Tigr *bmp = tigrBitmap(dw, dh);

  // No scaling
  if (dw == src->w && dh == src->h) {
    tigrBlit(bmp, src, 0, 0, 0, 0, dw, dh);
    return bmp;
  }

  for (int x = 0; x < dw; ++x) {
    for (int y = 0; y < dh; ++y) {
      double scaled_x = (double)x / dw * src->w;
      double scaled_y = (double)y / dh * src->h;

      int xl = (int)scaled_x, xu = MIN(xl + 1, src->w - 1);
      int yl = (int)scaled_y, yu = MIN(yl + 1, src->h - 1);

      double x_coeff1 = (xu - scaled_x), x_coeff2 = (scaled_x - xl);
      double y_coeff1 = (yu - scaled_y), y_coeff2 = (scaled_y - yl);

      bmp->pix[y * bmp->w + x] =
          (TPixel){y_coeff1 * (x_coeff1 * src->pix[yl * src->w + xl].r + x_coeff2 * src->pix[yl * src->w + xu].r) +
                       y_coeff2 * (x_coeff1 * src->pix[yu * src->w + xl].r + x_coeff2 * src->pix[yu * src->w + xu].r),
                   y_coeff1 * (x_coeff1 * src->pix[yl * src->w + xl].g + x_coeff2 * src->pix[yl * src->w + xu].g) +
                       y_coeff2 * (x_coeff1 * src->pix[yu * src->w + xl].g + x_coeff2 * src->pix[yu * src->w + xu].g),
                   y_coeff1 * (x_coeff1 * src->pix[yl * src->w + xl].b + x_coeff2 * src->pix[yl * src->w + xu].b) +
                       y_coeff2 * (x_coeff1 * src->pix[yu * src->w + xl].b + x_coeff2 * src->pix[yu * src->w + xu].b),
                   y_coeff1 * (x_coeff1 * src->pix[yl * src->w + xl].a + x_coeff2 * src->pix[yl * src->w + xu].a) +
                       y_coeff2 * (x_coeff1 * src->pix[yu * src->w + xl].a + x_coeff2 * src->pix[yu * src->w + xu].a)};
    }
  }
  return bmp;
}