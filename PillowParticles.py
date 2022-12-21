import numpy as np
import matplotlib.pyplot as plt
import cv2, os, random, math, glob
from PIL import Image

# CONSTANTS
# map of MAP_SIZE x MAP_SIZE pixels
MAP_SIZE = 100

# Blur magnitude
BLUR_STRENGTH = 3

# How fast trails fade to zero
DISSIPATE_TARGET = 200_000

# Number of agents
AGENTS_NUM = 1_000

# Max speed and sensor distance
SPEED = 1
LOOK_DISTANCE = 5

# Initial spread of agents (percent of MAP_SIZE)
CIRCLE_SIZE = 0.4

# Initialization
texture_map = None
agents = None

# Initalize scene
def init():
    global texture_map
    global agents
    
    # Create the texture map
    texture_map = np.random.random((MAP_SIZE, MAP_SIZE)) / 100
    
    # Update the circle size
    _circle = int(CIRCLE_SIZE * MAP_SIZE)

    # Give the agents some positions towards the middle
    agents = np.zeros((AGENTS_NUM, 4))
    for i in range(AGENTS_NUM):
        while True:
            x = random.randint(-_circle, _circle)
            y = random.randint(-_circle, _circle)
            if x ** 2 + y ** 2 > _circle**2:
                continue
            agents[i, 0] = x + MAP_SIZE / 2
            agents[i, 1] = x + MAP_SIZE / 2
            agents[i, 2] = (random.random() - 0.5) * 10
            agents[i, 3] = (random.random() - 0.5) * 10

            break
    
    _circle += np.random.rand(AGENTS_NUM, 4) * 0.1

    # Blurs the map and reduces the intesity of the map
    def diffuse_trails( _map ):
        _map += np.random.random((MAP_SIZE, MAP_SIZE)) / 4
        _blurred = cv2.GaussianBlur(_map, (BLUR_STRENGTH,BLUR_STRENGTH), 0)
        _blurred[_blurred > 20] = 20
        while _blurred.sum() > DISSIPATE_TARGET:
            _blurred *= 0.95
        _map[:,:] = _blurred
        return _blurred
    
    def update_agents( _map, _agents ):

        # Extract positions and velocities of each agent as ints
        integer_positions = np.rint(_agents[:,:2]).astype(int)
        integer_velocity = np.rint(_agents[:,2:]).astype(int)

        # Calculate look distance based off velocity
        forward_look = (integer_velocity * LOOK_DISTANCE)

        left_look = np.zeros((AGENTS_NUM, 2))
        left_look[:,0] = forward_look[:,1]
        left_look[:,1] = forward_look[:,0]

        
