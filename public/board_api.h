#ifndef BOARD_API_H
#define BOARD_API_H

#include "../include/_globals.h"

/** Ship ID enum
  - Used for indexing into the `remaining_ships` parameger of the `Board` struct (below)
*/
typedef enum ShipID : unsigned char {
  SHIP_PATROL_BOAT = 0,
  SHIP_SUBMARINE = 1,
  SHIP_DESTROYER = 2,
  SHIP_BATTLESHIP = 3,
  SHIP_CARRIER = 4
} ShipID;

/** Square enum
  - Provides the data for a given square on the board (details below)
*/
typedef enum Square : unsigned char {
  SQUARE_EMPTY = 0,
  SQUARE_MISS = 1,
  SQUARE_HIT = 2,
  SQUARE_SHIP_PATROL_BOAT = SHIP_PATROL_BOAT + SQUARE_HIT + 1,
  SQUARE_SHIP_SUBMARINE = SHIP_SUBMARINE + SQUARE_HIT + 1,
  SQUARE_SHIP_DESTROYER = SHIP_DESTROYER + SQUARE_HIT + 1,
  SQUARE_SHIP_BATTLESHIP = SHIP_BATTLESHIP + SQUARE_HIT + 1,
  SQUARE_SHIP_CARRIER = SHIP_CARRIER + SQUARE_HIT + 1
} Square;

/** Board struct
  - Provides the "player" view of the board

  @param board - 1D array representation of the 2D board - can be conveniently indexed into using the getSquare function
  below
    - For the players own board, shows hit/sunk markers, and also shows ships (full range of the `Square` enum)
    - For the opponent board, shows only hit/sunk markers (i.e. only `SQUARE_EMPTY`, `SQUARE_MISS`, `SQUARE_HIT` from
  the `Square` enum)
  @param remaining_ships - 1D array of the ships on the board - each ship's information is present at the index of its
  id. e.g. to retrieve data for the destroyer, get the value at `remaining_ships[SHIP_DESTROYER]`
    - For the players own board, this shows the number of remaining parts of the ship
    - For the opponent board, shows only whether the ship is still on the board (1) or has been sunk (0)
*/
typedef struct Board {
  Square board[BOARD_SIZE * BOARD_SIZE];
  char remaining_ships[(int)SHIP_CARRIER + 1];
} Board;

/** Coordinate struct
  - Provides xy coordinates on the board, with the top-left corner being (0,0), and coordinates increasing to the
  right(x) and downwards (y)
*/
typedef struct Coordinate {
  int x;
  int y;
} Coordinate;

/** Ship Placement struct
  - Provides xy coordinates and a rotation [0-3] for a ship placement
  - Rotation is counter-clockwise
*/
typedef struct ShipPlacement {
  int x;
  int y;
  int rotation;
} ShipPlacement;

/** Check ship function
  - Checks whether the given ship type can be placed at the provided location and rotation
  @param board - The board the ship should be placed on (passed to the user function)
  @param ship - The ship ID (passed to the user function)
  @param placement - The position and rotation of the ship (as returned by the user function)

  @returns int [0|1]
  - 0 | Invalid position
  - 1 | Valid position
*/
int checkShip(Board* board, ShipID ship, ShipPlacement placement);

/** Check ship function
  - Checks whether the given position has already been shot at
  @param board - The board the shot is being taken on (passed to the user function)
  @param shot - The position of the shot (as returned by the user function)

  @returns int [0|1]
  - 0 | Invalid position
  - 1 | Valid position
*/
int checkShot(Board* board, Coordinate shot);

/** Helper function for checking a board square
  - This can be done yourself, but this function provides an abstracted method
*/
static Square getSquare(Board* board, int x, int y) { return board->board[y * BOARD_SIZE + x]; }

#endif