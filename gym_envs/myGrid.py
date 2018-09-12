# This file is part of Bootstrapped Dual Policy Iteration
# 
# Copyright 2018, Vrije Universiteit Brussel (http://vub.ac.be)
#     authored by Hélène Plisnier <helene.plisnier@vub.be>
#
# BDPI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BDPI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BDPI.  If not, see <http://www.gnu.org/licenses/>.

import gym

from gym import spaces
from gym.utils import seeding

import random
import sys
import numpy as np


# Gridworld. ' ' is an empty cell, '-' or '|' is a wall, and 'G' the goal.
# 11 lines (y) and 9 rows (x), agent begins at the top left corner (GRID[0][0])
'''GRID = [
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ['-', '-', '-', '-', ' ', '-', '-', '-', '-'],
    [' ', ' ', '|', ' ', ' ', ' ', '|', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', '|', ' ', ' ', ' ', '|', ' ', ' '],
    ['-', '-', '-', '-', ' ', '-', '-', '-', '-'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'G'],
]
'''
REWARD = {' ': -1.0, 'G': 100.0}
TERMINAL = {' ': False,'G': True}

class myGrid(gym.Env):
    def __init__(self, y = 11, x = 9):
        num_states = x * y

        self.y = y
        self.x = x
        self.grid = [[' ' for x in range(self.x)] for y in range(self.y)]
        self.createGrid()
        self.display()
              
        self.action_space = spaces.Discrete(4)  # nb actions : Up, Down, Right, Left
        self.observation_space = spaces.Box(np.zeros((num_states,)), np.ones((num_states,)))   
        self._seed()
        
        self._max_x = 0
        self._max_y = 0
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def display(self):
        for y in range(self.y):
            for x in range(self.x):
                if x != self.x - 1:
                    print(self.grid[y][x], end='')
                else:
                    print(self.grid[y][x])
        

    def createGrid(self):
        roomHeight = (self.y - 2) // 3 # height of the gird - 2 walls divided by 3 rooms
        centerLine = (self.y - 1) // 2 # 2 of the 4 doors have this y coordinate
        centerColumn = (self.x - 1) // 2 # 2 of the 4 doors have this x coordinate

        z = 0
        for y in range(self.y):
            z += 1
            if (z > roomHeight):
                for x in range(self.x):
                    if not (x == centerColumn):
                        self.grid[y][x] = '-'
                z = 0
            if (y > roomHeight) and (y <= 2 *roomHeight) and (y != centerLine):
                for x in range(self.x):
                    if (((x+1) == roomHeight) or (x == 2*roomHeight)) :
                            self.grid[y][x] = '|'
                            
       # self.grid[self.x//3][(self.x-1)//2] = '#' # 4 doors
       # self.grid[(self.y-1)//2][(self.x-1)//3] = '#'
       # self.grid[(self.y-1)//2][(2*self.x)//3] = '#'
       # self.grid[((2*self.x)//3)+1][(self.x-1)//2] = '#'
        self.grid[self.y-1][self.x-1] = 'G'
        
        
    def displayPosition(self):
        for y in range(self.y):
            for x in range(self.x):
                if y == self._y and x == self._x:
                    if x != self.x-1:
                        print('X', end='')
                    else:
                        print('X')
                else:
                    if x != self.x-1:
                        print(self.grid[y][x], end='')
                    else:
                        print(self.grid[y][x])
        print('---------')
        print('---------')
                        
               
    def reset(self):
        """ Reset the environment and return the initial state number
        """
        print(self._max_x, self._max_y, file=sys.stderr)
        sys.stderr.flush()

        # Put the agent in the bottom-left corner of the environment
        self._x = 0
        self._y = 0
        self._timestep = 0
        self._max_x = 0
        self._max_y = 0

        return self.current_state()
        

    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:

            - 0: go up
            - 1: go down
            - 2: go left
            - 3: go right
        """
        assert(action >= 0)
        assert(action <= 3)
        
        self._timestep += 1
        self._max_x = max(self._max_x, self._x)
        self._max_y = max(self._max_y, self._y)

        # If cell resulting from action= '-' or '|'--> wall, then agent does not move
        wall_x = self._x
        wall_y = self._y
        
        if action == 0 and self._y > 0:
            # Go up
            self._y -= 1
        elif action == 1 and self._y < self.y - 1:
            # Go down
            self._y += 1
        elif action == 2 and self._x > 0:
            # Go left
            self._x -= 1
        elif action == 3 and self._x < self.x - 1:
            # Go right
            self._x += 1

        # wall
        if self.grid[self._y][self._x] == '-' or self.grid[self._y][self._x] == '|':
            self._x = wall_x
            self._y = wall_y
            
        cell = self.grid[self._y][self._x]
        if cell == 'G':
            print("Goal reached!")
        
        #self.displayPosition()
        
        # Return the current state, a reward and whether the episode terminates
        done = self._timestep > 500
        r = -50. if done else REWARD[cell]

        return self.current_state(), r, TERMINAL[cell] or done, {}

    def current_state(self):
        state_index = (self._y * self.x) + self._x
        num_states = self.x * self.y

        rs = np.zeros((num_states,), dtype=np.float32)
        rs[state_index] = 1.0

        return rs


