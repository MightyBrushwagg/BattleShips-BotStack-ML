#include "public/player1.h"

#include <stdlib.h>

#ifndef PYTHON_BOT
ShipPlacement player1PlaceShip(Board board, ShipID ship) {
  // return (ShipPlacement){-1, -1, -1};
  switch (ship) {
    case SHIP_PATROL_BOAT:
      return (ShipPlacement){1, 1, 0};
    case SHIP_SUBMARINE:
      return (ShipPlacement){3, 1, 0};
    case SHIP_DESTROYER:
      return (ShipPlacement){7, 2, 0};
    case SHIP_BATTLESHIP:
      return (ShipPlacement){1, 5, 0};
    case SHIP_CARRIER:
      return (ShipPlacement){5, 5, 0};
  }
}
Coordinate player1Action(Board board) {
  // return (Coordinate){-1, -1};
  int x, y;
  do {
    x = rand() % 10, y = rand() % 10;
  } while (!checkShot(&board, (Coordinate){x, y}));
  return (Coordinate){x, y};
}
#else

#define PLAYER_FILE "player1.py"

ShipPlacement player1PlaceShip(Board board, ShipID ship) {
  PyObject *args = PyTuple_New(3);
  PyObject *pyBoard = PyList_New(BOARD_SIZE * BOARD_SIZE);
  PyObject *pyRemainingShips = PyList_New(SHIP_CARRIER + 1);
  PyObject *pyShip = PyLong_FromLong((long)ship);
  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
    PyList_SET_ITEM(pyBoard, i, PyLong_FromLong((long)board.board[i]));
  }
  for (int i = 0; i < SHIP_CARRIER + 1; ++i) {
    PyList_SET_ITEM(pyRemainingShips, i, PyBool_FromLong((long)board.remaining_ships[i]));
  }
  PyTuple_SET_ITEM(args, 0, pyBoard), PyTuple_SET_ITEM(args, 1, pyRemainingShips), PyTuple_SET_ITEM(args, 2, pyShip);

  PyObject *placement = PyObject_CallObject(placeFnc, args);
  Py_DECREF(args), args = NULL;
  Py_DECREF(pyBoard), pyBoard = NULL;
  Py_DECREF(pyRemainingShips), pyRemainingShips = NULL;
  Py_DECREF(pyShip), pyShip = NULL;

  if (placement == NULL || !PyList_Check(placement) || PyList_GET_SIZE(placement) != 3) {
    if (placement != NULL)
      Py_DECREF(placement), placement = NULL;
    return (ShipPlacement){-1, -1, -1};
  }

  ShipPlacement result = (ShipPlacement){(int)PyLong_AsLong(PyList_GET_ITEM(placement, 0)),
                                         (int)PyLong_AsLong(PyList_GET_ITEM(placement, 1)),
                                         (int)PyLong_AsLong(PyList_GET_ITEM(placement, 2))};
  Py_DECREF(placement);
  return result;
}
Coordinate player1Action(Board board) {
  PyObject *args = PyTuple_New(2);
  PyObject *pyBoard = PyList_New(BOARD_SIZE * BOARD_SIZE);
  PyObject *pyRemainingShips = PyList_New(SHIP_CARRIER + 1);
  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
    PyList_SET_ITEM(pyBoard, i, PyLong_FromLong((long)board.board[i]));
  }
  for (int i = 0; i < SHIP_CARRIER + 1; ++i) {
    PyList_SET_ITEM(pyRemainingShips, i, PyBool_FromLong((long)board.remaining_ships[i]));
  }
  PyTuple_SET_ITEM(args, 0, pyBoard), PyTuple_SET_ITEM(args, 1, pyRemainingShips);

  PyObject *placement = PyObject_CallObject(actionFnc, args);
  Py_DECREF(args), args = NULL;
  Py_DECREF(pyBoard), pyBoard = NULL;
  Py_DECREF(pyRemainingShips), pyRemainingShips = NULL;

  if (placement == NULL || !PyList_Check(placement) || PyList_GET_SIZE(placement) != 2) {
    if (placement != NULL)
      Py_DECREF(placement), placement = NULL;
    return (Coordinate){-1, -1};
  }

  Coordinate result = (Coordinate){(int)PyLong_AsLong(PyList_GET_ITEM(placement, 0)),
                                   (int)PyLong_AsLong(PyList_GET_ITEM(placement, 1))};
  Py_DECREF(placement);
  return result;
}

#endif