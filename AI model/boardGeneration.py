import numpy as np
from time import sleep
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F

class Board():
    def __init__(self, X=10, Y=10, n = 10):
        self.X = X
        self.Y = Y
        self.n = n
        self.randomBoard(X=X, Y=Y)
        self.generateShots(X=X, Y=Y)
        self.shootShips()
        self.revealShips()


    # sets self.ship_placement
    def randomBoard(self, X=10, Y=10):

        empty_board = np.zeros((X,Y), dtype=int)
        # print(empty_board)

        # ship relative coordinates, which get messed with and used to check valid position
        # if valid, will be placed on board

        # ships kept in a numpy array
        # all coordinates relative to itself
        # x variables in top row
        # y variables in bottom row
        # for matrix multplication later on
        ships = [
        np.array([[0, 0], 
                [0, 1]]),
        np.array([[0, 0, 0],
                [0, 1, 2]]),
        np.array([[0, 0, 0, 1],
                [0, 1, 2, 0]]),
        np.array([[0, 0, 0, 0],
                [0, 1, 2, 3]]),
        np.array([[0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4]])
    ]
        counter = 1
        for ship_ in ships:
            # Get ship, place ship, test if valid, repeat if not
            place_valid = False
            while not place_valid:
                # sleep(5)
                # print("   ")
                # generate new coordinates
                new_x = np.random.randint(0,X-1)
                new_y = np.random.randint(0,Y-1)

                # permutations allowing rotation and flipping (flipping only has practical effect for the L shape)
                direction = np.random.randint(0,3)
                reflection = np.random.randint(0,1) > 0.5
                place_valid = True
                # print("Original", counter)
                # print(new_x, new_y, direction, reflection)
                # print(ship_)
                

                # rotate
                if direction == 0:
                    # up
                    # do nothing
                    rotated_ship = ship_
                    
                elif direction == 1:
                    # right
                    # rotate -90 degrees anticlockwise
                    theta = np.radians(-90)
                    c, s = np.cos(theta), np.sin(theta)
                    r_minus90 = np.array(((c, -s), (s, c)))
                    # print(r_minus90)
                    rotated_ship = np.matmul(r_minus90,ship_)
                    rotated_ship = np.round(rotated_ship)
                    
                elif direction == 2:
                    # down
                    # rotate 180 degrees
                    theta = np.radians(180)
                    c, s = np.cos(theta), np.sin(theta)
                    r_180 = np.array(((c, -s), (s, c)))
                    # print(r_180)
                    rotated_ship = np.matmul(r_180,ship_)
                    rotated_ship = np.round(rotated_ship)
                    
                elif direction == 3:
                    # left
                    # rotate 90 degrees clockwise
                    theta = np.radians(90)
                    c, s = np.cos(theta), np.sin(theta)
                    r_plus90 = np.array(((c, -s), (s, c)))
                    # print(r_plus90)
                    rotated_ship = np.matmul(r_plus90,ship_)
                    rotated_ship = np.round(rotated_ship)
                
                
                

                # reflection
                if reflection:
                    # flip left right (y axis)
                    flip_matrix = np.array(((-1, 0), (0, 1)))
                    permutated_ship = np.matmul(flip_matrix, rotated_ship)

                    
                else:
                    # no flip
                    permutated_ship = rotated_ship

                # print("Reflected")
                # print(permutated_ship)

                # add coordinates after rotation and reflections
                x_add = np.ones(len(permutated_ship[0]),dtype=int) * new_x
                y_add = np.ones(len(permutated_ship[0]),dtype=int) * new_y

                coord_add = np.concatenate((x_add,y_add))
                coord_add = coord_add.reshape(permutated_ship.shape)
                ship = np.add(permutated_ship, coord_add)

                # print("To place")
                ship = np.array(ship, dtype=int)
                # print(ship)


                for i in range(len(ship[0])):
                    # print(ship[0][i], ship[1][i])
                    if ship[0][i] < 0 or ship[1][i] < 0 or ship[0][i] >= X or ship[1][i] >= Y:
                        place_valid = False
                    elif empty_board[ship[0][i]][ship[1][i]] != 0:
                        place_valid = False

                    # place_valid = True

            for i in range(len(ship[0])):
                empty_board[ship[0][i]][ship[1][i]] = 1
            # print(empty_board)
            counter += 1
        
        # print("\n\n\n")
        # return empty_board
        self.ship_placement = empty_board

    # returns self.ship_placement
    def getShipPlacement(self):
        return self.ship_placement

    # sets self.shots
    def generateShots(self, X=10, Y=10):
        # Approached trying for now random matrix of hits 
        # this is where ML will be turnede average to great

        m = -1 + exp(2*self.n/(X*Y))
        # print(m)

        self.shots = 1*(np.random.normal(loc=m, scale = 0.2, size=(X,Y)) > 0.5)
        # print(hits)

    # returns self.shots
    def getShots(self):
        return self.shots

    # sets self.shot_board
    def shootShips(self, X=10, Y=10):
        if self.ship_placement.shape != self.shots.shape:
            print("Shapes don't match")
            raise ValueError
        
        self.hit_board = np.zeros((X,Y), dtype=float)
        
        for i in range(len(self.ship_placement)):
            for j in range(len(self.ship_placement[i])):
                if self.shots[i][j] == 1:
                    if self.ship_placement[i][j] == 1.0:
                        self.hit_board[i][j] = 1.0
                    else:
                        self.hit_board[i][j] = -1.0
                else:
                    self.hit_board[i][j] = 0.0

    # returns self.shot_board
    def getHits(self):
        return self.hit_board      

    # sets self.revealed_board
    def revealShips(self, X=10, Y=10):
        self.revealed_board = np.zeros((X,Y), dtype=float)

        for i in range(len(self.ship_placement)):
            for j in range(len(self.ship_placement[i])):
                if self.ship_placement[i][j] == 1.0:
                    self.revealed_board[i][j] = 1.0
                elif self.hit_board[i][j] == -1.0:
                    self.revealed_board[i][j] = -1.0
                else:
                    self.revealed_board[i][j] = 0.0

    # returns self.revealed_board
    def getRevealed(self):
        return self.revealed_board


    
# battleship = Board(n=10,X=10, Y=10)


# print("Ship placement \n{}".format(battleship.getShipPlacement()))

# print("       ")
# print("Shots \n{}".format(battleship.getShots()))


# print("     ")
# print("Hits \n{}".format(battleship.getHits()))

# print("     ")
# print("Revealed \n{}".format(battleship.getRevealed()))



