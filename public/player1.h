#include "board_api.h"

ShipPlacement player1PlaceShip(Board board, ShipID ship);
Coordinate player1Action(Board board);

#ifdef PYTHON_BOT
extern PyObject* placeFnc;
extern PyObject* actionFnc;
#endif