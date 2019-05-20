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
import math
import random
import sys
import numpy as np


# table height = width = from 0.0 to 1.0
# robot has 4 actions: turn left, right, go forward, backward

GOAL = np.array([0.5, 0.5, math.pi * 0.25])
TOLERANCE = np.array([0.05, 0.05, 0.3])

class Table(gym.Env):
    def __init__(self, rnd, backup):

        self.action_space = spaces.Discrete(3)  # 3 actions : turn left, right, go forward
        self.observation_space = spaces.Box(
            np.array([0.0, 0.0, -math.pi], dtype=np.float32),
            np.array([1.0, 1.0, math.pi], dtype=np.float32)
        )

        self.gui = None
        self.rnd = rnd
        self.backup = backup

        self._timestep = 0
        self.reset()

    def reset(self):
        """ Reset the environment and return the initial state number
        """
        if self.rnd:
            self._x = random.random()
            self._y = random.random()
            self._angle = random.uniform(-math.pi, math.pi)
        else:
            self._x = 0.1
            self._y = 0.1
            self._angle = 0.1

        self._timestep = 0
        #self.display()

        return self.current_state()

    def render(self):
        import gym_envs.table_gui

        if self.gui is None:
            self.gui = gym_envs.table_gui.TableGUI(GOAL[0], GOAL[1], GOAL[2])

        self.gui.display(self._x, self._y, self._angle)

    def step(self, action):
        """ Perform an action in the environment. Actions are as follows:

            - 0: turn left
            - 1: turn right
            - 2: go forward
            - 3: go backward
        """
        assert(action >= 0)
        assert(action <= 2)

        self._timestep += 1

        original_x = self._x
        original_y = self._y

        # Backup policy
        if self.backup:
            x, y, angle = self._x, self._y, self._angle
            p2 = math.pi / 2.
            turn = False

            if x < 0.1 and p2 < angle < p2*3:
                turn = True
            if x > 0.9 and not (p2 < angle < p2*3):
                turn = True
            if y < 0.1 and angle > math.pi:
                turn = True
            if y > 0.9 and angle < math.pi:
                turn = True

            if turn:
                action = 0

        # If cell resulting from action= '-' or '|'--> wall, then agent does not move
        if action == 0 :
            # turn left
            self._angle = (self._angle + 0.1) % (2 * math.pi)
        elif action == 1 :
            # turn right
            self._angle = (self._angle - 0.1) % (2 * math.pi)
        elif action == 2 :
            # go forward
            self._x += math.cos(self._angle) * 0.005
            self._y += math.sin(self._angle) * 0.005

        reward = 0.0
        terminal = False

        # if robot falls off table:
        if not (0.0 < self._x < 1.0 and 0.0 < self._y < 1.0):
            self._x = original_x
            self._y = original_y
            reward = -50
            terminal = True
            print("Fell off the table!")

        # Check goal
        state = self.current_state()

        if (np.abs(state - GOAL) < TOLERANCE).all():
            reward = 100
            terminal = True
            print("Goal reached!")


        # Return the current state, a reward and whether the episode terminates
        return state, reward, terminal or (self._timestep > 2000), {}

    def current_state(self):
        return np.array([self._x, self._y, self._angle], dtype=np.float32)


