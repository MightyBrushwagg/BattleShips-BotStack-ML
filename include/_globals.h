#ifndef GLOBALS
#define GLOBALS

// #define NO_GRAPHICS
// #define PLAYER
// #include<python3/Python.h>
#ifdef PYTHON_BOT
#define PY_SSIZE_T_CLEAN
#include <python3.11/Python.h>

#define PLAYER_FILE "player1.py"
#endif

#define BOARD_SIZE 10

#define BOARD_SQUARE_SIZE 50
#define BOARD_MARGIN_SIZE 5
#define BOARD_RENDER_SIZE ((BOARD_SIZE * BOARD_SQUARE_SIZE) + ((BOARD_SIZE - 1) * BOARD_MARGIN_SIZE))

#define TEXT_SPACE 30
#define ICON_SPACE 175

#define SHOT_R 10

#ifdef NO_FRAME_DELAY
#define FRAME_TIME 0
#else
#define FRAME_TIME 50  // 20 FPS
#endif

#define BG_COLOUR \
  (TPixel) { 0x0f, 0x0f, 0x0f, 0xff }
#define SQ_COLOUR \
  (TPixel) { 0x12, 0x96, 0xff, 0xff }
#define PLACEMENT_VALID_COLOUR \
  (TPixel) { 0xBf, 0xBf, 0xBf, 0xff }
#define PLACEMENT_INVALID_COLOUR \
  (TPixel) { 0xff, 0x00, 0x00, 0xff }
#define SHIP_COLOUR \
  (TPixel) { 0x8f, 0x8f, 0x8f, 0xff }

#define SHIP_DEAD_COLOUR \
  (TPixel) { 0x1f, 0x1f, 0x1f, 0xff }
#define SHIP_LIVE_COLOUR \
  (TPixel) { 0xff, 0xff, 0xff, 0xff }

typedef enum HitType { SHOT_FAIL = -1, SHOT_MISS = 0, SHOT_HIT = 1, SHOT_HIT_SUNK = 2 } HitType;

// Helper function for player input
static int toCoords(const char* input, int* x, int* y) {
  *x = -1, *y = -1;

  if (input[0] == '0')
    return 0;

  if (!(input[0] >= 'A' && input[0] <= 'K') && !(input[0] >= 'a' && input[0] <= 'k'))
    return 0;
  *x = (input[0] >= 'a' ? input[0] - 32 : input[0]) - 'A';

  for (int i = 1; i < 4; ++i) {
    if (input[i] == 0)
      return *y != -1;

    if (input[i] >= '0' && input[i] <= '9')
      *y = *y == -1 ? (input[i] - '0') : (*y * 10 + (input[i] - '0'));
  }
  return 0;
}

#endif