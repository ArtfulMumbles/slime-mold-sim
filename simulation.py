import math, random, pygame
import numpy as np

# Decay constant
DECAY_RATE = 0.1

# Diffusion constant
DIFFUSION_RATE = 0.92

# Screen size
SCREEN_X = 640
SCREEN_Y = 640

# Amount agent can randomly turn
ANGLE_BIAS = math.pi / 8

# Number of agents
N = 100

# True == draw grid
DRAW_GRID = True

# Size of grid
GRID_SIZE = 100

# Set the target frame rate
FPS = 144


class Agent:
    def __init__(self, x, y, orientation):
        # Set the position and orientation of the agent
        self.x = x
        self.y = y
        self.orientation = orientation

        # Set the agent's radius
        self.radius = int(200 / GRID_SIZE)

        # Set the agent's speed
        self.speed = 5

        # Set the agent's color
        self.color = (255, 255, 255)  # White

    def draw(self, screen):
        # Draw the agent on the screen
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def move(self):    
        
        # Sense for pheromones in front of agent
        #self.sense()
        
        # Update the position of the agent based on its orientation and speed
        self.x += math.cos(self.orientation) * self.speed
        self.y += math.sin(self.orientation) * self.speed

        # Wrap the agent around to the other side of the screen if it goes off the screen
        if self.x < 0:
            self.x = SCREEN_X
        elif self.x > SCREEN_X:
            self.x = 0
        if self.y < 0:
            self.y = SCREEN_Y
        elif self.y > SCREEN_Y:
            self.y = 0
            
    
    # Sense whether to turn left, right, or keep going straight
    def sense(self):
        
        x, y = self.current_grid()
        
        neighbors = []
        
        if self.orientation > math.pi / 4 and self.orientation <= 3 * math.pi / 4:
            neighbors = [(access(x + 1, y + 1)), (access(x - 1, y + 1))]
        elif self.orientation > 3 * math.pi / 4 and self.orientation <= 5 * math.pi / 4:
            neighbors = [(access(x - 1 , y + 1)), (access(x - 1, y - 1))]
        elif self.orientation > 5 * math.pi / 4 and self.orientation <= 7 * math.pi / 4:
            neighbors = [(access(x + 1, y - 1)), (access(x - 1, y - 1))]
        else:
            neighbors = [(access(x + 1, y + 1)), (access(x + 1, y - 1))]

        
        d_theta = sum([neighbors[0], -1 * neighbors[1]])
        
        if d_theta == 0:
            return
        elif d_theta > 0:
            self.orientation += random.uniform(-ANGLE_BIAS, 0)
        else:
            self.orientation += random.uniform(0, ANGLE_BIAS)
            
        self.orientation %= math.pi * 2
                
            
    
            
    # Returns the agent's current grid square
    def current_grid(self):
        
        # Get the current grid square of an agent
        x = math.floor(self.x / (SCREEN_X / GRID_SIZE))
        y = math.floor(self.y / (SCREEN_Y / GRID_SIZE))
        
        # Wrap the x and y values around to the other side of the grid if they exceed the bounds of the grid
        x %= GRID_SIZE
        y %= GRID_SIZE
        
        return (x, y)

# Wrap around grid values
def access(x, y):
    if x < 0:
        x = GRID_SIZE - 1
    elif x >= GRID_SIZE:
        x = 0
    if y < 0:
        y = GRID_SIZE - 1
    elif y >= GRID_SIZE:
        y = 0
    return grid[x][y]

# Create a function to generate a heatmap color based on a value
def heatmap_color(value):
    # Set the minimum and maximum values for the heatmap
    min_value = 0
    max_value = 100

    # Normalize the value to a range between 0 and 1
    normalized_value = (value - min_value) / (max_value - min_value)

    # Map the normalized value to a range between 0 and 255
    color_value = int(normalized_value * 255)

    # Return a color tuple based on the heatmap value
    return (255, min(color_value, 255), 0)
    
# Initialize Pygame
pygame.init()

# Create the window
screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))

# Create a grid to store the pheromone values
grid = [[0 for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

# Set the initial pheromone value at the center of the grid
center_x = math.floor(GRID_SIZE / 2)
center_y = math.floor(GRID_SIZE / 2)
grid[center_x][center_y] = 50
grid[center_x + 1][center_y + 1] = 75

# Create a list to store the agents
agents = []

# Create n agents with random positions, radii, and orientations
for i in range(N):
    x = random.randint(0, 640)
    y = random.randint(0, 480)
    orientation = random.uniform(0, 2 * math.pi)
    agents.append(Agent(x, y, orientation))

# Create a clock object to control the frame rate
clock = pygame.time.Clock()

# Run the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Fill the window with white
    screen.fill((0, 0, 0))

    # Iterate over the grid and set the colors of the pixels on the screen based on the pheromone values
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            value = grid[x][y]
            if DRAW_GRID:
                color = heatmap_color(value)
                screen_x, screen_y = SCREEN_X, SCREEN_Y
                screen_x = x * screen_x / GRID_SIZE
                screen_y = y * screen_y / GRID_SIZE
                pygame.draw.rect(screen, color, (screen_x, screen_y, 64, 48))
            
            # Calculate the average pheromone value of the 8 neighboring grid squares
            neighbors = [            grid[(x - 1) % GRID_SIZE][(y - 1) % GRID_SIZE],
                grid[(x - 1) % GRID_SIZE][y],
                grid[(x - 1) % GRID_SIZE][(y + 1) % GRID_SIZE],
                grid[x][(y - 1) % GRID_SIZE],
                grid[x][(y + 1) % GRID_SIZE],
                grid[(x + 1) % GRID_SIZE][(y - 1) % GRID_SIZE],
                grid[(x + 1) % GRID_SIZE][y],
                grid[(x + 1) % GRID_SIZE][(y + 1) % GRID_SIZE],
            ]
            avg_pheromone = max(neighbors)
        
            # Calculate the new pheromone value for the current grid square
            new_pheromone = (1 - DECAY_RATE) * grid[x][y] + (1 - DIFFUSION_RATE) * avg_pheromone
            
            grid[x][y] = int(new_pheromone)

    # Update and draw all the agents
    for agent in agents:
        agent.move()
        agent.draw(screen)
        x, y = agent.current_grid()
        if (grid[x][y] < 100):
            grid[x][y] = 100

    # Update the display
    pygame.display.update()

    # Limit the frame rate to the target fps
    clock.tick(FPS)

# Quit Pygame
pygame.quit()

