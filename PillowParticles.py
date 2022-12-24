import numpy as np
import matplotlib.pyplot as plt
import cv2, os, random, math, glob
from PIL import Image

# CONSTANTS
# Number of iterations per simulation
ITERATIONS = 1000

# Only show map every this many frames
RENDER_SPEED = 1

# map of MAP_SIZE x MAP_SIZE pixels
MAP_SIZE = 20

# Blur magnitude
BLUR_STRENGTH = 3

# How fast trails fade to zero
DISSIPATE_TARGET = 200_000

# Number of agents
AGENTS_NUM = 10

# Max speed and sensor distance
SPEED = 1
LOOK_DISTANCE = 9

# Strength of trails left by agents
TRAIL_STRENGTH = 1

# Time steps per frame
STEP_SIZE = 1

# How much changes in velocity affect agent
NUDGE_STRENGTH = 3

# Initial spread of agents (percent of MAP_SIZE)
CIRCLE_SIZE = 0.6

# Initialization
texture_map = None
agents = None

# Initalize scene
def init ():
    global texture_map
    global agents

    # Create the map
    texture_map = np.random.random((MAP_SIZE, MAP_SIZE))/100

    # Update circle size
    _circle = int(CIRCLE_SIZE * MAP_SIZE)

    # Give the agents some positions towards the middle
    agents = np.zeros((AGENTS_NUM, 4))
    for i in range(AGENTS_NUM):
        while True:
            x = random.randint(-_circle, _circle)
            y = random.randint(-_circle, _circle)
            if x ** 2 + y ** 2 > _circle ** 2:
                continue
            agents[i, 0] = x + MAP_SIZE / 2
            agents[i, 1] = y + MAP_SIZE / 2
            agents[i, 2] = (random.random() - 0.5) * 10
            agents[i, 3] = (random.random() - 0.5) * 10
            
            break
    agents += np.random.rand(AGENTS_NUM, 4) * 0.1

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
    left_look[:,1] = -forward_look[:,0]
    
    right_look = np.zeros((AGENTS_NUM, 2))
    right_look[:,0] = -forward_look[:,1]
    right_look[:,1] = forward_look[:,0]
    
    forward_cast = (integer_positions + forward_look).astype(int)
    forward_cast[forward_cast > MAP_SIZE - 1] = MAP_SIZE - 1
    forward_cast[forward_cast < 0] = 0
    strength_forward = _map[forward_cast[:,0], forward_cast[:, 1]]
    
    left_cast = (integer_positions + left_look).astype(int)
    left_cast[left_cast > MAP_SIZE - 1] = MAP_SIZE - 1
    left_cast[left_cast < 0] = 0
    strength_left = _map[left_cast[:,0], left_cast[:,1]]
    
    right_cast = (integer_positions + right_look).astype(int)
    right_cast[right_cast > MAP_SIZE - 1] = MAP_SIZE - 1
    right_cast[right_cast < 0] = 0
    strength_right = _map[right_cast[:,0], right_cast[:,1]]
    
    # Compute the weighted average of the strengths
    total_strength = strength_forward + strength_left + strength_right
    
    percentage_forward = strength_forward / total_strength
    percentage_left = strength_left / total_strength
    percentage_right = strength_right / total_strength
    
    # Target velocity
    invert_vel = np.zeros((AGENTS_NUM, 2))
    invert_vel[:,0] = _agents[:,3]
    invert_vel[:,1] = -_agents[:,2]
    
    target_vel = np.zeros((AGENTS_NUM, 2))
    
    target_vel[:,0] += _agents[:,2] * percentage_forward
    target_vel[:,0] += invert_vel[:,0] * percentage_left
    target_vel[:,0] -= invert_vel[:,0] * percentage_right

    target_vel[:,1] += _agents[:,3] * percentage_forward
    target_vel[:,1] += invert_vel[:,1] * percentage_left
    target_vel[:,1] -= invert_vel[:,1] * percentage_right
    
    _agents[:,2:] += target_vel * STEP_SIZE * NUDGE_STRENGTH
    
    _velocity = np.sqrt(_agents[:,2] ** 2 + agents[:,3] ** 2)
    _agents[:,2] *= SPEED/_velocity
    _agents[:,3] *= SPEED/_velocity
    
    _agents[:,0] += _agents[:,2] * STEP_SIZE
    _agents[:,1] += _agents[:,3] * STEP_SIZE
    
    # Make sure all agents are in bounds
    _x_oob = np.any([_agents[:,0] <= 0, _agents[:,0] >= MAP_SIZE - 1], axis=0)
    _y_oob = np.any([_agents[:,1] <= 0, _agents[:,1] >= MAP_SIZE - 1], axis=0)
    
    _agents[:,0] = np.clip(_agents[:,0], 0, MAP_SIZE-1)
    _agents[:,1] = np.clip(_agents[:,1], 0, MAP_SIZE-1)
    
    # Bounce off the walls
    _agents[:,2][_x_oob] *= np.random.random((1))[0] / 3 - 1.17
    _agents[:,3][_y_oob] *= np.random.random((1))[0] / 3 - 1.17
    
# Generates trails on the map
def trails(_map, _agents):
    integer_positions = np.rint(_agents[:,:2]).astype(int)
    _map[integer_positions[:,0], integer_positions[:,1]] += TRAIL_STRENGTH

# TODO: Figure out this code!!!
def tag_agents(_map, _agents, offset):
    _intensity = 1
    integer_positions = np.rint(_agents[:,:2]).astype(int)
    integer_positions += offset

# A single step of the simulation
def simulation_step(_map, _agents):
    diffuse_trails(_map)
    update_agents(_map, _agents)
    trails(_map, _agents)
    
# Renders information to the screen
def show_map(m, save, do):
    # Copy the map to new array and pad
    _map = np.zeros((MAP_SIZE + 4, MAP_SIZE + 4))
    _map[2:-2, 2:-2] = m
    
    # Scale the map
    _map = np.log10(_map)
    _map = _map / _map.max() * 255
    _map[0,0] = 255
    
    _colored = np.zeros((MAP_SIZE+4, MAP_SIZE+4, 3))
    _colored[:,:,1] = 0
    _colored[:,:,2] = 0
    _colored[:,:,0] = _map
    _colored[_colored < 0] = - _colored[_colored < 0]
    _colored[_colored > 255] = 255 - _colored[_colored > 255]
    _colored[_colored < 0] = 0
    
    '''
    _colored = np.zeros((MAP_SIZE+4, MAP_SIZE+4, 3))
    _colored[:,:,1] = _map
    _colored[:,:,2] = (_map - 127) * 2
    _colored[:,:,0] = 0
    _colored[_colored < 0] = -_colored[_colored < 0]
    _colored[_colored > 255] = 255 - _colored[_colored > 255]
    _colored[_colored < 0] = 0
    '''
    _colored = cv2.GaussianBlur(_colored,(BLUR_STRENGTH, BLUR_STRENGTH),0)
    
    _colored = _colored.astype(np.uint8)
    
    if not do: return
    
    # Render the map
    
    im = Image.fromarray(_colored)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(f'results/{save[0]}/saves/{save[1]}.png')
    
def run_simulation():
    
    global texture_map 
    
    run_id = str(random.randint(0,100000))
    os.mkdir(f'results/{run_id}')
    os.mkdir(f'results/{run_id}/saves')
    print(f'Saving simulation as {run_id}')
    
    # Initialize
    init()
    
    # Run the simulation
    for i in range(ITERATIONS):
        # Calculate the progress as a percentage
        progress = (i+1) / ITERATIONS

        # Calculate the number of "=" characters to use in the progress bar
        num_equals = int(progress * 30)

        # Create the progress bar string
        progress_bar = '=' * num_equals + '-' * (30 - num_equals)

        # Print the progress bar and overwrite the previous output
        print(f'[{progress_bar}] {round(progress * 100, 1)}%', end='\r')
        simulation_step(texture_map, agents)
        show_map(texture_map, (run_id, str.rjust(str(i), 4, '0')), i % RENDER_SPEED == 0)
        
    return run_id

def compile_results(_id):
    #Load all of the images
    images = glob.glob(f"results/{_id}/saves/*.png")
    images.sort()
    
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(f"results/{_id}/out.gif", format="GIF", append_images=frames,
                   save_all=True, duration=20, loop=0)
    
sim_id = run_simulation()
compile_results(sim_id)
print(f'Simulation results saved in: {sim_id}')
            
        
