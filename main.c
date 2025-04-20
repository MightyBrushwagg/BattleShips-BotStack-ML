// Included first due to the python header needing to be first
#include "include/_globals.h"
// These are included as part of the Python include, so only need to be included if this isn't being used
#ifndef PYTHON_BOT
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#endif

// Game logic headers
#include "include/board.h"

// Rendering headers
#ifndef NO_GRAPHICS
#include "include/tigr_resize.h"
#endif

// Player headers
#include "public/player1.h"
#include "public/player2.h"

#define ERROR(msg)        \
  printf("[ERROR] " msg); \
  return 1;

ShipData ships[5] = {
    (ShipData){1, 2, 0, 0, 0x8080000000000000},
    (ShipData){1, 3, 0, 1, 0x8080800000000000},
    (ShipData){2, 3, 1, 1, 0x8080C00000000000},
    (ShipData){1, 4, 0, 2, 0x8080808000000000},
    (ShipData){1, 5, 0, 3, 0x8080808080000000}
};
#ifndef NO_GRAPHICS
TigrFont* arial;

int shootAnim(Tigr* game, BoardData* b, int player1, int x, int y);

void updateGameRender(Tigr* game, BoardData* p1, BoardData* p2);
void updateGameOverlays(Tigr* game);
void renderLabels(Tigr* game);
void renderIcons(Tigr* game);
void renderDeath(Tigr* game, int i);
#endif

static Coordinate toRender(int x, int y, int board) {
  return (Coordinate){
      TEXT_SPACE + x * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE) + BOARD_SQUARE_SIZE / 2 +
          (board == 2) * (BOARD_RENDER_SIZE + TEXT_SPACE + BOARD_SQUARE_SIZE),
      TEXT_SPACE + y * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE) + BOARD_SQUARE_SIZE / 2,
  };
}

#define BUFFER 4

#define LOG_LOCATION stdout

#define PLAYER_1     0b00000001
#define SHIPS_PLACED 0b00000010
#define FIRST_RENDER 0b00000100
#define GAME_OVER    0b00001000

AnimationState anim_state = ANIMATION_NONE;
int anim_time = 0;
Coordinate anim_pos;

#ifdef PYTHON_BOT
PyObject* placeFnc;
PyObject* actionFnc;
#endif

int main() {
#ifdef PYTHON_BOT
  // Initialise python and add the current directory to sys.path
  Py_Initialize();
  PyObject* sysmodule = PyImport_ImportModule("sys");
  PyObject* syspath = PyObject_GetAttrString(sysmodule, "path");
  PyList_Append(syspath, PyBytes_FromString("."));
  Py_DECREF(syspath), Py_DECREF(sysmodule);

  FILE* exp_file;
  PyObject *main_module, *global_dict;
  // Load the relevant functions from the file
  exp_file = fopen(PLAYER_FILE, "r");
  PyRun_SimpleFile(exp_file, PLAYER_FILE);

  main_module = PyImport_AddModule("__main__");
  global_dict = PyModule_GetDict(main_module);

  char* exec_str = calloc(300, sizeof(char));
  sprintf(exec_str,
          "BOARD_SIZE=%d\nSHIP_PATROL_BOAT=%d\nSHIP_SUBMARINE=%d\nSHIP_DESTROYER=%d\nSHIP_BATTLESHIP="
          "%d\nSHIP_CARRIER=%d\nSQUARE_EMPTY=%d\nSQUARE_MISS=%d\nSQUARE_HIT=%d\nSQUARE_SHIP_PATROL_BOAT="
          "%d\nSQUARE_SHIP_SUBMARINE=%d\nSQUARE_SHIP_DESTROYER=%d\nSQUARE_SHIP_BATTLESHIP="
          "%d\nSQUARE_SHIP_CARRIER=%d\n",
          BOARD_SIZE, SHIP_PATROL_BOAT, SHIP_SUBMARINE, SHIP_DESTROYER, SHIP_BATTLESHIP, SHIP_CARRIER, SQUARE_EMPTY,
          SQUARE_MISS, SQUARE_HIT, SQUARE_SHIP_PATROL_BOAT, SQUARE_SHIP_SUBMARINE, SQUARE_SHIP_DESTROYER,
          SQUARE_SHIP_BATTLESHIP, SQUARE_SHIP_CARRIER);
  PyRun_SimpleString(exec_str);
  free(exec_str);

  placeFnc = PyDict_GetItemString(global_dict, "player1PlaceShip");
  actionFnc = PyDict_GetItemString(global_dict, "player1Action");
  if (actionFnc == NULL || placeFnc == NULL) {
    ERROR("Failed to load Python functions")
  }
#endif

  srand(time(0));
#ifndef NO_GRAPHICS
  // ---------- Load resources ----------
  Tigr* arial_img = tigrLoadImage("resources/arial.png");
  if (arial_img == NULL) {
    ERROR("Failed to load font image")
  }
  arial = tigrLoadFont(arial_img, TCP_1252);
  if (arial == NULL) {
    ERROR("Failed to load font")
  }

  // ---------- Window setup ----------
  Tigr* game = tigrBitmap(2 * BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
                          TEXT_SPACE + BOARD_RENDER_SIZE + ICON_SPACE);
  tigrClear(game, BG_COLOUR);
  game->blitMode = TIGR_KEEP_ALPHA;
  renderLabels(game);
  renderIcons(game);

  TigrResize* window = (TigrResize*)calloc(1, sizeof(TigrResize));
  window->window = tigrWindow(game->w, game->h, "Game", TIGR_AUTO);
  window->contents = game;
#endif

  // ---------- Game variables ----------
  // Gamestate
  unsigned char flags = PLAYER_1;
#ifdef PLAYER
  ShipID playerCurrentShip = SHIP_PATROL_BOAT;
#endif

  // Function pointers
  ShipPlacement (*p1PlaceFuncPtr)(Board, ShipID) = &player1PlaceShip;
  Coordinate (*p1ActionFuncPtr)(Board) = &player1Action;
  ShipPlacement (*p2PlaceFuncPtr)(Board, ShipID) = &player2PlaceShip;
  Coordinate (*p2ActionFuncPtr)(Board) = &player2Action;

  // ---------- Player setup ----------
  BoardData* p1 = initBoardData(0);
  BoardData* p2 = initBoardData(1);

  ShipPlacement placement;
  Board* b;
  Coordinate shot;
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    b = toBoard(p1, 0);
    placement = (*p1PlaceFuncPtr)(*b, i);
    free(b);
    if (placement.x == -1 || placement.y == -1 || placement.rotation == -1) {
      ERROR("Player 1 - Undefined ship placement function")
    }
    if (!placeShip(p1, i, placement.x, placement.y, placement.rotation)) {
      ERROR("Player 1 - Invalid ship placement")
    }
#ifdef NO_GRAPHICS
    else
      fprintf(LOG_LOCATION, "[Player 1] Place %d | %d %d - %d\n", i, placement.x, placement.y, placement.rotation);
#endif
  }

  char s[BUFFER];
  int x, y, str_ptr = 0, c, sunk;
  memset(s, 0, BUFFER);

#ifndef PLAYER
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    b = toBoard(p2, 0);
    placement = (*p2PlaceFuncPtr)(*b, i);
    free(b);
    if (placement.x == -1 || placement.y == -1 || placement.rotation == -1) {
      ERROR("Player 2 - Undefined ship placement function")
    }
    if (!placeShip(p2, i, placement.x, placement.y, placement.rotation)) {
      ERROR("Player 2 - Invalid ship placement")
    }
#ifdef NO_GRAPHICS
    else
      fprintf(LOG_LOCATION, "[Player 2] Place %d | %d %d - %d\n", i, placement.x, placement.y, placement.rotation);
#endif
  }
  flags |= SHIPS_PLACED;
#elif defined(NO_GRAPHICS)
  int rotation;
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    printf("Enter the position for ship %d: ", i + 1);
    c = 0;
    fseek(stdin, 0, SEEK_END);
    while (c != '\n' && c != '\r') {
      c = getchar();
      s[str_ptr++] = c;
      if (str_ptr == BUFFER) {
        memset(s, 0, BUFFER);
        str_ptr = 0;
      }
    }
    toCoords(s, &x, &y);
    if (x == -1 || y == -1) {
      printf("Invalid input\n");
      --i;
      memset(s, 0, BUFFER);
      str_ptr = 0;
      continue;
    }
    printf("Enter the rotation [0-3]: ");
    fseek(stdin, 0, SEEK_END);
    c = getchar();
    if (c < '0' || c > '3') {
      printf("Invalid input\n");
      --i;
    }
    else {
      rotation = c - '0';
      if (!placeShip(p2, i, x, y, rotation)) {
        printf("Invalid ship placement\n");
        --i;
      }
      else
        fprintf(LOG_LOCATION, "[Player 2] Place %d | %d %d - %d\n", i, x, y, rotation);
    }

    memset(s, 0, BUFFER);
    str_ptr = 0;
  }
#else
  placement = (ShipPlacement){5, 5, 0};
#endif

  clock_t t, last_t = clock();
  while (
#ifndef NO_GRAPHICS
      !tigrClosed(window->window)
#else
      1
#endif
  ) {
    t = clock();
    if (t - last_t < FRAME_TIME)
      continue;
    last_t = t;

    // Manage player manual ship placement
#if defined(PLAYER) && !defined(NO_GRAPHICS)
    if (!(flags & SHIPS_PLACED)) {
      if (!(flags & FIRST_RENDER)) {
        flags |= FIRST_RENDER;

        updateGameRender(game, p1, p2);
        renderPlacementOverlay(p2, game, 0, playerCurrentShip, placement.x, placement.y, placement.rotation);
      }
      c = tigrReadChar(window->window);
      if (c != 0) {
        switch (c) {
          case 'a':
            if (placement.x > 0)
              --placement.x;
            break;
          case 'd':
            if (placement.x < BOARD_SIZE - 1)
              ++placement.x;
            break;
          case 'w':
            if (placement.y > 0)
              --placement.y;
            break;
          case 's':
            if (placement.y < BOARD_SIZE - 1)
              ++placement.y;
            break;
          case ' ':
            placement.rotation = (placement.rotation + 1) % 4;
            break;
          case '\n':
          case '\r':
            if (placeShip(p2, playerCurrentShip, placement.x, placement.y, placement.rotation) == 1) {
              playerCurrentShip++;
              placement = (ShipPlacement){5, 5, 0};
              if (playerCurrentShip > SHIP_CARRIER)
                flags |= SHIPS_PLACED;
            }
            break;
        }
        updateGameRender(game, p1, p2);
        if (playerCurrentShip <= SHIP_CARRIER)
          renderPlacementOverlay(p2, game, 0, playerCurrentShip, placement.x, placement.y, placement.rotation);
      }
      tigrResizeUpdate(window);
      continue;
    }
#endif

    if (anim_state == ANIMATION_NONE) {
      if (flags & PLAYER_1) {
        b = toBoard(p2, 1);
        shot = (*p1ActionFuncPtr)(*b);
        if (shot.x == -1 && shot.y == -1) {
          ERROR("Player 1 - Unimplemented shot function")
        }
#ifdef NO_GRAPHICS
        fprintf(LOG_LOCATION, "[Player 1] Shoot   | %d %d\n", shot.x, shot.y);
#endif
        if (
#ifndef NO_GRAPHICS
            !shootAnim(game, p2, 0, shot.x, shot.y)
#else
            !shoot(p2, shot.x, shot.y, &sunk)
#endif
        )
          flags &= ~PLAYER_1;
        if (p2->remaining_ships == 0) {
          printf("Player 1 wins\n");
          flags |= GAME_OVER;
        }
      }
      else {
#if defined(PLAYER) && !defined(NO_GRAPHICS)
        c = tigrReadChar(window->window);
        if (c != 0) {
          if (c == '\n' || c == '\r') {
            s[str_ptr] = 0;
            if (toCoords(s, &x, &y)) {
              if (!shootAnim(game, p1, 1, shot.x, shot.y))
                flags |= PLAYER_1;
              if (p1->remaining_ships == 0) {
                printf("Player 2 wins\n");
                flags |= GAME_OVER;
              }
            }
            memset(s, 0, BUFFER);
            str_ptr = 0;
          }
          else {
            s[str_ptr++] = c;
            if (str_ptr == BUFFER) {
              memset(s, 0, BUFFER);
              str_ptr = 0;
            }
          }
        }
#else
        b = toBoard(p1, 1);
        shot = (*p2ActionFuncPtr)(*b);
        if (shot.x == -1 && shot.y == -1) {
          ERROR("Player 2 - Unimplemented shot function")
        }
#ifdef NO_GRAPHICS
        fprintf(LOG_LOCATION, "[Player 2] Shoot   | %d %d\n", shot.x, shot.y);
#endif
        if (
#ifndef NO_GRAPHICS
            !shootAnim(game, p1, 1, shot.x, shot.y)
#else
            !shoot(p1, shot.x, shot.y, &sunk)
#endif
        )
          flags |= PLAYER_1;
        if (p1->remaining_ships == 0) {
          printf("Player 2 wins\n");
          flags |= GAME_OVER;
        }
#endif
      }
    }

#ifndef NO_GRAPHICS
    updateGameRender(game, p1, p2);
    if (anim_state != ANIMATION_NONE)
      updateGameOverlays(game);
    tigrResizeUpdate(window);
    if (anim_state == ANIMATION_NONE && flags & GAME_OVER)
      break;
#else
    if (flags & GAME_OVER)
      break;
#endif
  }

#ifndef NO_GRAPHICS
  tigrFree(game);
  tigrFree(window->window);
  if (window->contents_display != NULL)
    tigrFree(window->contents_display);
  free(window);
#endif

#ifdef PYTHON_BOT
  if (Py_FinalizeEx() < 0) {
    exit(120);
  }
  fclose(exp_file);
#endif

  return 0;
}

#ifndef NO_GRAPHICS
int shootAnim(Tigr* game, BoardData* b, int player1, int x, int y) {
  int sunk;
  switch (shoot(b, x, y, &sunk)) {
    case SHOT_FAIL:
      return 0;
    case SHOT_HIT:
      anim_state = ANIMATION_HIT;
      anim_pos = toRender(x, y, player1 ? 1 : 2);
      return 1;
    case SHOT_HIT_SUNK:
      anim_state = ANIMATION_HIT;
      anim_pos = toRender(x, y, player1 ? 1 : 2);
      renderDeath(game, (player1 ? 0 : 5) + (sunk + 1));
      return 1;
    case SHOT_MISS:
      anim_state = ANIMATION_SPLASH;
      anim_pos = toRender(x, y, player1 ? 1 : 2);
      return 0;
  }
}

void updateGameRender(Tigr* game, BoardData* p1, BoardData* p2) {
  tigrBlit(game, p1->board_render, TEXT_SPACE, TEXT_SPACE, 0, 0, BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);
  tigrBlit(game, p2->board_render, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE, TEXT_SPACE, 0, 0,
           BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);
}
void updateGameOverlays(Tigr* game) {
  if (anim_state == ANIMATION_SPLASH) {
    if (anim_time < ANIMATION_DURATION_SPLASH_TRANSITION) {
      int r = ease_in(0, ANIMATION_DURATION_SPLASH_TRANSITION, SHOT_R, (int)((BOARD_SQUARE_SIZE - 1) / 2), anim_time);
      tigrFill(game, anim_pos.x - BOARD_SQUARE_SIZE / 2, anim_pos.y - BOARD_SQUARE_SIZE / 2, BOARD_SQUARE_SIZE,
               BOARD_SQUARE_SIZE,
               tigrGet(game, anim_pos.x - BOARD_SQUARE_SIZE / 2, anim_pos.y - BOARD_SQUARE_SIZE / 2));
      tigrFillCircle(game, anim_pos.x, anim_pos.y, r, (TPixel){0x00, 0x00, 0x00, 0xff});
    }
    else {
      int r = ease_out(ANIMATION_DURATION_SPLASH_TRANSITION, ANIMATION_DURATION_SPLASH, SHOT_R,
                       (int)((BOARD_SQUARE_SIZE - 1) / 2), anim_time);
      tigrCircle(game, anim_pos.x, anim_pos.y, r - 2, COLOUR_SPLASH_2);
      tigrCircle(game, anim_pos.x, anim_pos.y, r - 1, COLOUR_SPLASH_1);
      tigrCircle(game, anim_pos.x, anim_pos.y, r, (TPixel){0xff, 0xff, 0xff, 0xff});
    }

    if (anim_time++ == ANIMATION_DURATION_SPLASH) {
      anim_time = 0;
      anim_state = ANIMATION_NONE;
    }
  }
  else if (anim_state == ANIMATION_HIT) {
    if (anim_time < ANIMATION_DURATION_SPLASH_TRANSITION) {
      int r = ease_in(0, ANIMATION_DURATION_HIT_TRANSITION, SHOT_R, (int)((BOARD_SQUARE_SIZE - 1) / 2), anim_time);
      tigrFill(game, anim_pos.x - BOARD_SQUARE_SIZE / 2, anim_pos.y - BOARD_SQUARE_SIZE / 2, BOARD_SQUARE_SIZE,
               BOARD_SQUARE_SIZE,
               tigrGet(game, anim_pos.x - BOARD_SQUARE_SIZE / 2, anim_pos.y - BOARD_SQUARE_SIZE / 2));
      tigrFillCircle(game, anim_pos.x, anim_pos.y, r, (TPixel){0x00, 0x00, 0x00, 0xff});
    }
    else {
      int r = ease_out(ANIMATION_DURATION_HIT_TRANSITION, ANIMATION_DURATION_HIT, SHOT_R,
                       (int)((BOARD_SQUARE_SIZE - 1) / 2), anim_time);
      tigrCircle(game, anim_pos.x, anim_pos.y, r - 1, COLOUR_HIT_1);
      tigrCircle(game, anim_pos.x, anim_pos.y, r, COLOUR_HIT_2);
    }
    if (anim_time++ == ANIMATION_DURATION_HIT) {
      anim_time = 0;
      anim_state = ANIMATION_NONE;
    }
  }
}
void renderLabels(Tigr* game) {
  char s[BUFFER];
  int x, y;
  for (int i = 0; i < BOARD_SIZE; ++i) {
    // Row labels
    sprintf(s, "%d", i);
    x = tigrTextWidth(arial, s), y = tigrTextHeight(arial, s);
    tigrPrint(game, arial, (TEXT_SPACE - x) / 2,
              TEXT_SPACE + i * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE) + (BOARD_SQUARE_SIZE - y) / 2,
              (TPixel){0xff, 0xff, 0xff, 0xff}, s);
    tigrPrint(game, arial, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + TEXT_SPACE + (TEXT_SPACE - x) / 2,
              TEXT_SPACE + i * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE) + (BOARD_SQUARE_SIZE - y) / 2,
              (TPixel){0xff, 0xff, 0xff, 0xff}, s);

    // Column labels
    sprintf(s, "%c", i + 65);
    x = tigrTextWidth(arial, s), y = tigrTextHeight(arial, s);
    tigrPrint(game, arial, TEXT_SPACE + i * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE) + (BOARD_SQUARE_SIZE - x) / 2,
              (TEXT_SPACE - y) / 2, (TPixel){0xff, 0xff, 0xff, 0xff}, s);
    tigrPrint(game, arial,
              BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE + i * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE) +
                  (BOARD_SQUARE_SIZE - x) / 2,
              (TEXT_SPACE - y) / 2, (TPixel){0xff, 0xff, 0xff, 0xff}, s);
  }
}
void renderIcons(Tigr* game) {
  Tigr* icon = tigrLoadImage("resources/Gunboat.png");
  tigrBlitTint(game, icon, TEXT_SPACE, TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h,
               SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Destroyer.png");
  tigrBlitTint(game, icon, TEXT_SPACE + 3 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
               TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Submarine.png");
  tigrBlitTint(game, icon, TEXT_SPACE + 7 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
               TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Battleship.png");
  tigrBlitTint(game, icon, TEXT_SPACE, TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h,
               SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Carrier.png");
  tigrBlitTint(game, icon, TEXT_SPACE + 5 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
               TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Gunboat.png");
  tigrBlitTint(game, icon, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
               TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Destroyer.png");
  tigrBlitTint(game, icon,
               BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE + 3 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
               TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Submarine.png");
  tigrBlitTint(game, icon,
               BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE + 7 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
               TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Battleship.png");
  tigrBlitTint(game, icon, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
               TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);

  icon = tigrLoadImage("resources/Carrier.png");
  tigrBlitTint(game, icon,
               BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE + 5 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
               TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h, SHIP_LIVE_COLOUR);
  tigrFree(icon);
}
void renderDeath(Tigr* game, int i) {
  static Tigr* icon;
  switch (i) {
    case 1:
      icon = tigrLoadImage("resources/Gunboat.png");
      tigrBlitTint(game, icon, TEXT_SPACE, TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h,
                   SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 2:
      icon = tigrLoadImage("resources/Submarine.png");
      tigrBlitTint(game, icon, TEXT_SPACE + 7 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
                   TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 3:
      icon = tigrLoadImage("resources/Destroyer.png");
      tigrBlitTint(game, icon, TEXT_SPACE + 3 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
                   TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 4:
      icon = tigrLoadImage("resources/Battleship.png");
      tigrBlitTint(game, icon, TEXT_SPACE, TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h,
                   SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 5:
      icon = tigrLoadImage("resources/Carrier.png");
      tigrBlitTint(game, icon, TEXT_SPACE + 5 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
                   TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 6:
      icon = tigrLoadImage("resources/Gunboat.png");
      tigrBlitTint(game, icon, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
                   TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 7:
      icon = tigrLoadImage("resources/Submarine.png");
      tigrBlitTint(game, icon,
                   BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE + 7 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
                   TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 8:
      icon = tigrLoadImage("resources/Destroyer.png");
      tigrBlitTint(game, icon,
                   BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE + 3 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
                   TEXT_SPACE + BOARD_RENDER_SIZE + 75 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 9:
      icon = tigrLoadImage("resources/Battleship.png");
      tigrBlitTint(game, icon, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
                   TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
    case 10:
      icon = tigrLoadImage("resources/Carrier.png");
      tigrBlitTint(game, icon,
                   BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE + 5 * (BOARD_SQUARE_SIZE + BOARD_MARGIN_SIZE),
                   TEXT_SPACE + BOARD_RENDER_SIZE + 150 - icon->h, 0, 0, icon->w, icon->h, SHIP_DEAD_COLOUR);
      tigrFree(icon);
      break;
  }
}
#endif