import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import sleep

'''
### Key Implementation Notes ###

Don't touch the function definitions for 'player1PlaceShip' and 'player1Action' - it will crash the program
The data passed to these functions is essentially the same as for the C implementation, except using lists rather than arrays

The C macro/enum constants are provided automatically by the main program, such as ship and square ids, and the board size


### Important details ###

For an unknown reason, modification of global variables (variables defined outside the scope of a function) from inside 
 functions will crash the program. If you want to preserve variables between function calls, which also being able to modify
 them, use function attributes 
 (a helper decorator has been provided for you - example is in+above the default `player1Action` function)


### Linking PYTHON to C via gcc

You may need to have the environment variables PYTHONPATH and PYTHONHOME set, depending on your installation of python
The program will fault if this is the case, and these variables should be set to the parent directory of the directory where your 
 'python.exe' file is located

You may well also need to change the include in _globals.h - my installation of python is 3.10, so the current include is
  `#include <python3.10/Python.h>`
 but this may need to change depending on your installation.

'''


'''
Game constants - set automatically when the script is integrated into the C environment
- They exist here only so they are valid variables according to python syntax highlighters
'''
SHIP_PATROL_BOAT = SHIP_SUBMARINE = SHIP_DESTROYER = SHIP_BATTLESHIP = SHIP_CARRIER = 0
SQUARE_EMPTY = SQUARE_MISS = SQUARE_HIT = SQUARE_SHIP_PATROL_BOAT = SQUARE_SHIP_SUBMARINE = SQUARE_SHIP_DESTROYER = SQUARE_SHIP_BATTLESHIP = SQUARE_SHIP_CARRIER = 0

BOARD_SIZE = 0

''' 
Helper function for querying the board list at given coordinates rather than directly indexing
'''
def getBoardSquare(board: list, x : int, y : int):
  return board[y + x*BOARD_SIZE]

'''
Helper decorator used for "static" variables
'''
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def player1PlaceShip(board : list, remaining_ships : list, ship : int):
  if ship == SHIP_PATROL_BOAT:
    return [1,1,0]
  if ship == SHIP_SUBMARINE:
    return [3, 1, 0]
  if ship == SHIP_DESTROYER:
    return [7, 2, 0]
  if ship == SHIP_BATTLESHIP:
    return [1, 5, 0]
  if ship == SHIP_CARRIER:
    return [5, 5, 0]
  return [-1, -1, -1]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, padding=1) # output is shape: [8, 9, 9]
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, padding=1) # output is shape: [16, 8, 8]
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, padding=1) # output is shape: [32, 7, 7]
        self.pool = nn.MaxPool2d(2, 2) # output shape is: [32, 3, 3]

        self.fc1 = nn.Linear(256, 150)
        self.fc2 = nn.Linear(150, 84)
        self.fc3 = nn.Linear(84, 100)  # 100 = 10x10 ship placement prediction

    def forward(self, x):
        print("Hello")
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        # x = F.tanh(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flattens so can then be put in fully connected network
        # print(x.shape)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)  # No activation (MSELoss expects raw output)
        return x

@static_vars(x = -1)
@static_vars(y = 0)
def player1Action(board: list, remaining_ships : list):

    board_input = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=float)

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            square = getBoardSquare(board, j, i)
            if square == 1:
                board_input[i][j] = -1.0
            elif square == 2:
                board_input[i][j] = 1.0

    # --- Model prediction ---
    model = Net()
    # model.load_state_dict(torch.load(r"/Users/xavierparker/Desktop/Bot Stack/battleships/AI model/trained.pth"))
    model.load_state_dict(torch.load(r"/Users/xavierparker/Desktop/Bot Stack/battleships/best models/twomillion.pth"))
    model.eval()

    input_tensor = torch.from_numpy(board_input).unsqueeze(0).unsqueeze(0).float()  # shape: [1, 1, 10, 10]

    with torch.no_grad():
        output = model(input_tensor)  # output shape: [1, 100]
        output = output.view(BOARD_SIZE, BOARD_SIZE)  # reshapes output to [10, 10] for analysis

    #  locate where to shoot
    max_x = 0
    max_y = 0
    max_val = -float('inf')

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board_input[i][j] == 0 and output[i][j] > max_val:
                max_val = output[i][j]
                max_x = j
                max_y = i


    print("   ")
    print(f"Best move: x={max_x}, y={max_y}, predicted score={max_val:.4f}")
    print(board_input)
    print(output)


    # sleep(10)

    return[max_x,max_y]


    # if player1Action.x == BOARD_SIZE:
    #   player1Action.x = -1
    #   player1Action.y += 1
    # if player1Action.y == BOARD_SIZE:
    #   return [1, 1]
    # player1Action.x += 1
    # return [1, 1]