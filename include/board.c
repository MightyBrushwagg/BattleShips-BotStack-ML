#include "board.h"

#include <stdio.h>

#define DISP_0 0x8000000000000000

static void rotate(int *x, int *y, int rot) {
  int temp;
  switch (rot) {
    case 0:  // No rotation
      break;
    case 1:  // 90 ccw
      temp = *x;
      *x = *y;
      *y = -temp;
      break;
    case 2:  // 180 ccw
      *x = -*x;
      *y = -*y;
      break;
    case 3:  // 270 ccw
      temp = *x;
      *x = -*y;
      *y = temp;
      break;
    default:  // Other
      break;
  }
}
static void rotate2(int *x, int *y, int rot) {
  int temp;
  switch (rot) {
    case 0:  // No rotation
      break;
    case 1:  // 90 ccw
      temp = *x;
      *x = -*y;
      *y = temp;
      break;
    case 2:  // 180 ccw
      *x = -*x;
      *y = -*y;
      break;
    case 3:  // 270 ccw
      temp = *x;
      *x = *y;
      *y = -temp;
      break;
    default:  // Other
      break;
  }
}

BoardData *initBoardData(int primaryPlayer) {
  BoardData *b = (BoardData *)calloc(1, sizeof(BoardData));
  b->primaryPlayer = primaryPlayer;
#ifndef NO_GRAPHICS
  b->board_render = tigrBitmap(BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);
  tigrClear(b->board_render, BG_COLOUR);

  // Initialise the grid render
  for (int i = 0; i < BOARD_SIZE; ++i) {
    for (int j = 0; j < BOARD_SIZE; ++j) {
      tigrFill(b->board_render, i * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
               j * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE), BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE, SQ_COLOUR);
    }
  }
#endif

  return b;
}

#ifndef NO_GRAPHICS
void updateSquare(Tigr *board, int primaryPlayer, int x, int y, Square s) {
  x *= (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE);
  y *= (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE);
  TPixel c;
  switch (s) {
    case SQUARE_EMPTY:
      tigrFill(board, x, y, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE, SQ_COLOUR);
      break;
    case SQUARE_SHIP_PATROL_BOAT:
    case SQUARE_SHIP_SUBMARINE:
    case SQUARE_SHIP_DESTROYER:
    case SQUARE_SHIP_BATTLESHIP:
    case SQUARE_SHIP_CARRIER:
      c = (TPixel){0x5f + 25 * (s - SQUARE_SHIP_PATROL_BOAT), 0x5f + 25 * (s - SQUARE_SHIP_PATROL_BOAT),
                   0x5f + 25 * (s - SQUARE_SHIP_PATROL_BOAT), 0xff};
#if defined(PLAYER) && !defined(SPECTATOR)
      tigrFill(board, x, y, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE, primaryPlayer ? c : SQ_COLOUR);
#else
      tigrFill(board, x, y, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE, c);
#endif
      break;
    case SQUARE_HIT:
      tigrFillCircle(board, x + BOARD_SQUARE_SIZE / 2, y + BOARD_SQUARE_SIZE / 2, SHOT_R,
                     (TPixel){0xff, 0x00, 0x00, 0xff});
      break;
    case SQUARE_MISS:
      tigrFillCircle(board, x + BOARD_SQUARE_SIZE / 2, y + BOARD_SQUARE_SIZE / 2, SHOT_R,
                     (TPixel){0x00, 0x00, 0x00, 0xff});
      break;
  }
}
#endif

int placeShip(BoardData *board, ShipID shipId, int x, int y, int rot) {
  if (board == NULL)
    return 0;

  ShipData ship = ships[shipId];

  int x_place = -ship.w + ship.offset_x + 1, y_place = -ship.h + ship.offset_y + 1;
  rotate(&x_place, &y_place, rot);
  x_place += x, y_place += y;
  int w_rot = ship.w - 1, h_rot = ship.h - 1;
  rotate(&w_rot, &h_rot, rot);

  if (OUT_OF_BOUNDS_WH(x_place, y_place, w_rot, h_rot))
    return 0;

  Coordinate *coords = calloc(ship.w * ship.h, sizeof(Coordinate));
  int ptr = -1;

  int i_rot, j_rot;
  for (int i = 0; i < ship.h; ++i) {
    for (int j = 0; j < ship.w; ++j) {
      if (ship.disp & (DISP_0 >> (j + i * 8))) {
        i_rot = i, j_rot = j;
        rotate2(&i_rot, &j_rot, rot);

        if (board->board[(y_place + i_rot) + (x_place + j_rot) * BOARD_SIZE] == SQUARE_EMPTY)
          coords[++ptr] = (Coordinate){(x_place + j_rot), (y_place + i_rot)};
        else {
          free(coords);
          return 0;
        }
      }
    }
  }
  board->remaining_ship_squares[shipId] = ptr + 1;
  board->remaining_ships++;
  for (; ptr >= 0; --ptr) {
    board->board[coords[ptr].y + coords[ptr].x * BOARD_SIZE] = SQUARE_SHIP_PATROL_BOAT + shipId;
#ifndef NO_GRAPHICS
    updateSquare(board->board_render, board->primaryPlayer, coords[ptr].x, coords[ptr].y,
                 shipId + SQUARE_SHIP_PATROL_BOAT);
#endif
  }
  free(coords);

  return 1;
}
HitType shoot(BoardData *board, int x, int y, int *sunk) {
  *sunk = -1;
  if (OUT_OF_BOUNDS(x, y))
    return SHOT_FAIL;

  int i = y + x * BOARD_SIZE;
  switch (board->board[i]) {
    case SQUARE_HIT:
    case SQUARE_MISS:
      return SHOT_FAIL;
    case SQUARE_EMPTY:
      board->board[i] = SQUARE_MISS;
#ifndef NO_GRAPHICS
      updateSquare(board->board_render, board->primaryPlayer, x, y, SQUARE_MISS);
#endif
      return SHOT_MISS;
    case SQUARE_SHIP_PATROL_BOAT:
    case SQUARE_SHIP_SUBMARINE:
    case SQUARE_SHIP_DESTROYER:
    case SQUARE_SHIP_BATTLESHIP:
    case SQUARE_SHIP_CARRIER:
      board->remaining_ship_squares[board->board[i] - SQUARE_SHIP_PATROL_BOAT] -= 1;
#ifndef NO_GRAPHICS
      updateSquare(board->board_render, board->primaryPlayer, x, y, SQUARE_HIT);
#endif

      if (board->remaining_ship_squares[board->board[i] - SQUARE_SHIP_PATROL_BOAT] == 0) {
        board->remaining_ships -= 1;
        *sunk = board->board[i] - SQUARE_SHIP_PATROL_BOAT;
        board->board[i] = SQUARE_HIT;
        return SHOT_HIT_SUNK;
      }
      board->board[i] = SQUARE_HIT;
      return SHOT_HIT;
  }
}

#ifndef NO_GRAPHICS
void renderPlacementOverlay(BoardData *board, Tigr *render, int player1, ShipID shipId, int x, int y, int rot) {
  if (board == NULL)
    return;

  ShipData ship = ships[shipId];

  int x_place = -ship.w + ship.offset_x + 1, y_place = -ship.h + ship.offset_y + 1;
  rotate(&x_place, &y_place, rot);
  x_place += x, y_place += y;
  int w_rot = ship.w - 1, h_rot = ship.h - 1;
  rotate(&w_rot, &h_rot, rot);

  Coordinate *coords = calloc(ship.w * ship.h, sizeof(Coordinate));
  int ptr = -1, valid = 1;

  int i_rot, j_rot;
  for (int i = 0; i < ship.h; ++i) {
    for (int j = 0; j < ship.w; ++j) {
      if (ship.disp & (DISP_0 >> (j + i * 8))) {
        i_rot = i, j_rot = j;
        rotate2(&i_rot, &j_rot, rot);

        coords[++ptr] = (Coordinate){(x_place + j_rot), (y_place + i_rot)};
        if (board->board[(y_place + i_rot) + (x_place + j_rot) * BOARD_SIZE] != SQUARE_EMPTY)
          valid = 0;
        if (OUT_OF_BOUNDS(y_place + i_rot, x_place + j_rot))
          valid = 0;
      }
    }
  }
  board->remaining_ship_squares[shipId] = ptr + 1;
  board->remaining_ships += 1;
  for (; ptr >= 0; --ptr) {
    if (OUT_OF_BOUNDS(coords[ptr].x, coords[ptr].y))
      continue;
    int render_x = TEXT_SPACE + coords[ptr].x * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE);
    int render_y = TEXT_SPACE + coords[ptr].y * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE);

    if (!player1)
      render_x += BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + TEXT_SPACE;
    tigrFill(render, render_x, render_y, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE,
             valid ? PLACEMENT_VALID_COLOUR : PLACEMENT_INVALID_COLOUR);
  }
  free(coords);
}
#endif

// -------------------- API functions --------------------
int checkShip(Board *board, ShipID shipID, ShipPlacement placement) {
  if (board == NULL || shipID > SHIP_CARRIER)
    return 0;

  ShipData ship = ships[shipID];

  int x_place = -ship.w + ship.offset_x + 1, y_place = -ship.h + ship.offset_y + 1;
  rotate(&x_place, &y_place, placement.rotation);
  x_place += placement.x, y_place += placement.y;
  int w_rot = ship.w - 1, h_rot = ship.h - 1;
  rotate(&w_rot, &h_rot, placement.rotation);

  if (OUT_OF_BOUNDS_WH(x_place, y_place, w_rot, h_rot))
    return 0;

  int i_rot, j_rot;
  for (int i = 0; i < ship.h; ++i) {
    for (int j = 0; j < ship.w; ++j) {
      if (ship.disp & (DISP_0 >> (j + i * 8))) {
        i_rot = i, j_rot = j;
        rotate2(&i_rot, &j_rot, placement.rotation);

        if (board->board[(y_place + i_rot) + (x_place + j_rot) * BOARD_SIZE] != SQUARE_EMPTY)
          return 0;
      }
    }
  }
  return 1;
}
int checkShot(Board *board, Coordinate shot) {
  if (OUT_OF_BOUNDS(shot.x, shot.y))
    return 0;
  return board->board[shot.y + shot.x * BOARD_SIZE] != SQUARE_HIT &&
         board->board[shot.y + shot.x * BOARD_SIZE] != SQUARE_MISS;
}