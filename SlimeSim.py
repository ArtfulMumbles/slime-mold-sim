'''
TODO: Random ordering in motor and sensory stages.
TODO: Add sensor size greater than one
'''

import numpy as np
import random, glob, time, cv2

from PIL import Image
from os import mkdir

# SIMULATION PARAMETERS
# Number of steps
NUM_STEPS = 5000

# How often to render frames
RENDER_SPEED = 2

# MODEL PARAMETERS
# Map size in pixels
MAP_SIZE = 1200

# Number of agents as percentage of map area
NUM_AGENTS = int(MAP_SIZE ** 2 * 0.1)

# Size of starting circle as percentage of map size
START_CIRCLE = 0.3

# Distance moved each time step
STEP_SIZE = 1

# Time passed per excecution step
TIME_STEP = 1

# Amount of attractant deposited per timestep
DEPOSITION_RATE = 5

# View distance of agent
SENSOR_OFFSET = 9

# Size of diffusion kernal
DIFFUSION_KERNAL = 3

# Rate of attractant decay
DECAY_RATE = 0.1

# Sensor angle from forward position
SENSOR_ANGLE = np.deg2rad(45)

# How far agents rotate
ROTATION_ANGLE = np.deg2rad(45)

# Define pheromone mask
trail_map = None

# Define agents list
# Shape: (AGENTS_NUM, 4)
# First two columns are x and y position
# Second two columns are x and y velocities
agents = None

# Initalize simulation
def init_sim():
    global agents, trail_map

    # Initialize trail_map
    trail_map = np.zeros((MAP_SIZE, MAP_SIZE))

    # Starting circle radius
    _circle = int(MAP_SIZE * START_CIRCLE)

    # Initialize agents
    agents = np.zeros((NUM_AGENTS, 3))

    # Initalize agents
    for i in range(NUM_AGENTS):
        while(True):
            # Randomly generate agent within circle
            x = random.randint(-_circle, _circle)
            y = random.randint(-_circle, _circle)
            if x ** 2 + y ** 2 > _circle ** 2:
                continue;
            agents[i, 0] = x + MAP_SIZE / 2;
            agents[i, 1] = y + MAP_SIZE / 2;

            # Give agents random orientations
            # Generate random angle 0 to 2pi
            agents[i, 2] = 2 * np.pi * np.random.random()
            break;
        
# Update positions of agents and restrict them back in bounds if necessary.
#  If an agent is within bounds, deposit attractant at its position on the map.
def motor_stage(_agents, _map):
    # Update x and y coordinates of agents using numpy's cos and sin functions
    _agents[:, 0] += STEP_SIZE * np.cos(_agents[:, 2]) * TIME_STEP
    _agents[:, 1] += STEP_SIZE * np.sin(_agents[:, 2]) * TIME_STEP

    # Check if any agents are out of bounds
    out_of_bounds = (_agents[:, 0] < 0) | (_agents[:, 0] > MAP_SIZE - 1) | (_agents[:, 1] < 0) | (_agents[:, 1] > MAP_SIZE - 1)

    # Restrict out-of-bounds agents back in bounds and give them new, random angles
    _agents[out_of_bounds, 0] = np.clip(_agents[out_of_bounds, 0], 0, MAP_SIZE - 1)
    _agents[out_of_bounds, 1] = np.clip(_agents[out_of_bounds, 1], 0, MAP_SIZE - 1)
    _agents[out_of_bounds, 2] = np.random.random(size=out_of_bounds.sum()) * 2 * np.pi

    # Only deposit attractant at rounded locations of in-bounds agents
    in_bounds = ~out_of_bounds
    _map[np.rint(_agents[in_bounds, 0]).astype(int), np.rint(_agents[in_bounds, 1]).astype(int)] += DEPOSITION_RATE * TIME_STEP

def sensor_stage(_agents, _map):
    # Calculate sensor weights in front of each agent
    forward_weights = sense(_agents, _map, 0)
    left_weights = sense(_agents, _map, SENSOR_ANGLE)
    right_weights = sense(_agents, _map, -SENSOR_ANGLE)

    # Generate random steer factors
    random_steers = np.random.uniform(low=0, high=1.2, size=NUM_AGENTS)

    # Change orientation of agents based on sensor weights
    change = np.zeros(NUM_AGENTS)
    change[(forward_weights < left_weights) &
     (forward_weights < right_weights)] = (random_steers[(forward_weights < left_weights) & 
     (forward_weights < right_weights)] - 0.5) * 2 * ROTATION_ANGLE * TIME_STEP
    change[(left_weights < right_weights)] -= random_steers[(left_weights < right_weights)] * ROTATION_ANGLE * TIME_STEP
    change[(right_weights < left_weights)] += random_steers[(right_weights < left_weights)] * ROTATION_ANGLE * TIME_STEP
    _agents[:, 2] += change


        

def sense(_agents, _map, _sensor_angle_offset):
    # Calculate sensor offset distances
    sensor_offset_x = np.cos(_agents[:, 2] + _sensor_angle_offset) * SENSOR_OFFSET
    sensor_offset_y = np.sin(_agents[:, 2] + _sensor_angle_offset) * SENSOR_OFFSET

    # Determine sensor locations and ensure they are within bounds
    sensor_x = np.clip(_agents[:, 0] + sensor_offset_x, 0, MAP_SIZE - 1).astype(int)
    sensor_y = np.clip(_agents[:, 1] + sensor_offset_y, 0, MAP_SIZE - 1).astype(int)

    # Return sensed values
    return _map[sensor_x, sensor_y]


def update_trails(_map):
    _map = cv2.GaussianBlur(_map, (DIFFUSION_KERNAL, DIFFUSION_KERNAL), 0)
    _map = np.multiply(_map, 1.0 - DECAY_RATE) 


def draw_map(_map, _id, save):
    _colored = np.zeros((MAP_SIZE + 4, MAP_SIZE + 4))
    _colored[2:-2, 2:-2] = _map

    _colored = np.log(_colored + 1)

    _colored = np.interp(_colored, (_colored.min(), _colored.max()), (0, 255))
    _colored = np.around(_colored).astype(np.uint8)

    image = Image.fromarray(_colored)
    image.save(f"results/{_id}/saves/{str.rjust(str(save), 4, '0')}.png")

def draw_agents(_agents, _id, save):
    _colored = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    for x, y in _agents[:,:2]:
        _colored[int(y), int(x)] = [255, 255, 255]
    image = Image.fromarray(_colored)
    image.save(f"results/{_id}/saves/{str.rjust(str(save), 4, '0')}.png")

def simulate(type=True):
    global agents, trail_map

    run_id = str(random.randint(0,10000))
    mkdir(f'results/{run_id}')
    mkdir(f'results/{run_id}/saves')
    print(f'Saving simulation as {run_id}')

    # Initialize simulation
    init_sim()

    total_time = 0

    for i in range(NUM_STEPS):
        # Timer
        start_time = time.time()
        # Progress bar
        # Calculate the progress as a percentage
        progress = (i+1) / NUM_STEPS

        # Calculate the number of "=" characters to use in the progress bar
        num_equals = int(progress * 30)

        # Create the progress bar string
        progress_bar = '=' * num_equals + '-' * (30 - num_equals)

        # Print the progress bar and overwrite the previous output
        print(f' eta: {round((total_time / (i + 1)) * (NUM_STEPS - i), 0)} seconds',
        f'[{progress_bar}] {round(progress * 100, 1)}%',
        end='\r')


        # Simulation steps
        update_trails(trail_map)
        motor_stage(agents, trail_map)
        sensor_stage(agents, trail_map)

        # Save image of model state
        if i % RENDER_SPEED == 0:
            if type:
                draw_agents(agents, run_id, i + 1)
            else:
                draw_map(trail_map, run_id, i + 1)
        
        # Timer
        total_time += time.time() - start_time

    # Compile results
    compile_results(run_id)

def compile_results(_id):
    #Load all of the images
    images = glob.glob(f"results/{_id}/saves/*.png")
    images.sort()
    
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(f"results/{_id}/out.gif", format="GIF", append_images=frames,
                   save_all=True, duration=20, loop=0)



start_time = time.time()
simulate(True)
print("\r")
print("--- %s seconds ---" % round(time.time() - start_time, 2))



# Motor stage
