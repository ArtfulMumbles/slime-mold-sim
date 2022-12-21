import numpy as np
import matplotlib.pyplot as plt
import cv2, os, random, math, glob
from PIL import Image

# CONSTANTS
# map of MAP_SIZE x MAP_SIZE pixels
MAP_SIZE = 100

# Initial spread of agents
CIRCLE_SIZE

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
    
    
