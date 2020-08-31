#!/bin/usr/env python3

import numpy as np

WIDTH = 5
HEIGHT = 5

DEPTH = 80
GAMMA = 0.9

def calc_value(values, rewards, visited, x, y, depth, dir=''):

    if visited[x,y] == depth:
        return

    visited[x,y] += 1

    value = 0

    if rewards[x,y] != 0:
        if x == 0 and y == 1:
            calc_value(values, rewards, visited, 4, 1, depth)
            value = values[4,1]

        elif x == 0 and y == 3:
            calc_value(values, rewards, visited, 2, 3, depth)
            value = values[2,3]

    else:

        height = values.shape[0]
        width  = values.shape[1]

        # north
        if dir != 's':
            if x == 0:
                value += -0.25
            else:
                calc_value(values, rewards, visited, x-1, y, depth, 'n')
                value += 0.25*values[x-1,y]

        # south
        if dir != 'n':
            if x == (height-1):
                value += -0.25
            else:
                calc_value(values, rewards, visited, x+1, y, depth, 's')
                value += 0.25*values[x+1,y]

        # east
        if dir != 'w':
            if y == (width-1):
                value += -0.25
            else:
                calc_value(values, rewards, visited, x, y+1, depth, 'e')
                value += 0.25*values[x,y+1]

        # west
        if dir != 'e':
            if y == 0:
                value += -0.25
            else:
                calc_value(values, rewards, visited, x, y-1, depth, 'w')
                value += 0.25*values[x,y-1]

    values[x,y] = rewards[x,y] + (GAMMA * value)

values = np.zeros((HEIGHT, WIDTH))
rewards = np.zeros((HEIGHT, WIDTH))
visited = np.zeros((HEIGHT, WIDTH))

# special states
rewards[0,1] = 10.0
rewards[0,3] = 5.0

init_x = 4
init_y = 4

calc_value(values, rewards, visited, init_x, init_y, DEPTH)

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
print(values)
