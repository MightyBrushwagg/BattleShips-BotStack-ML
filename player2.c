#include "public/player2.h"

#include <stdlib.h>

ShipPlacement player2PlaceShip(Board board, ShipID ship) {
  // return (ShipPlacement){-1, -1, -1};
  switch (ship) {
    case SHIP_PATROL_BOAT: // 1x2
      return (ShipPlacement){5, 3, 0};
    case SHIP_SUBMARINE: // 
      return (ShipPlacement){0, 1, 0};
    case SHIP_DESTROYER:
      return (ShipPlacement){2, 4, 1};
    case SHIP_BATTLESHIP:
      return (ShipPlacement){5, 8, 3};
    case SHIP_CARRIER:
      return (ShipPlacement){5, 6, 3};
  }
}
Coordinate player2Action(Board board) {
  // return (Coordinate){-1, -1};
  int x, y;
  do {
    x = rand() % 10, y = rand() % 10;
  } while (!checkShot(&board, (Coordinate){x, y}));
  return (Coordinate){x, y};
}