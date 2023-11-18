import pygame
import sys
import time
from smart_home_env import SmartHomeTempControlEnv

# Define colors
WHITE = (255, 255, 255)
BLUE = (0, 128, 255)
RED = (255, 100, 70)
GREEN = (0, 255, 127)
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)

class Graph:
    def __init__(self, screen, env, width, height, x, y):
        self.screen = screen
        self.env = env
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.temp_history = [env.reset()[0]] * (width // 2)  # Initialize with starting temperature
        self.font = pygame.font.Font('freesansbold.ttf', 24)
        self.gridline_count = 10
        self.max_temp_value = 50 
        self.min_temp_value = 0
        self.optimal_temp_value = env.user_preference

    def update(self, current_temp):
        self.temp_history.pop(0)
        self.temp_history.append(current_temp)
        self.draw()

    def draw(self):
        # Draw border and gridlines
        pygame.draw.rect(self.screen, BLACK, (self.x, self.y, self.width, self.height), 3)
        for i in range(self.gridline_count + 1):
            line_y = self.y + i * self.height / self.gridline_count
            pygame.draw.line(self.screen, LIGHT_GRAY, (self.x, line_y), (self.x + self.width, line_y))

        # Draw the graph
        step_size = self.width / len(self.temp_history)
        for i in range(1, len(self.temp_history)):
            from_x = self.x + (i - 1) * step_size
            to_x = self.x + i * step_size
            from_temp_normalized = (self.temp_history[i - 1] - self.min_temp_value) / (self.max_temp_value - self.min_temp_value)
            to_temp_normalized = (self.temp_history[i] - self.min_temp_value) / (self.max_temp_value - self.min_temp_value)
            from_y = self.y + self.height - (from_temp_normalized * self.height)
            to_y = self.y + self.height - (to_temp_normalized * self.height)
            pygame.draw.line(self.screen, GREEN, (from_x, from_y), (to_x, to_y), 3)
        
        # Draw optimal temperature line
        optimal_temp_normalized = (self.optimal_temp_value - self.min_temp_value) / (self.max_temp_value - self.min_temp_value)
        optimal_temp_y = self.y + self.height - (optimal_temp_normalized * self.height)
        pygame.draw.line(self.screen, BLUE, (self.x, optimal_temp_y), (self.x + self.width, optimal_temp_y), 3)

        # Labels and title
        for i in range(self.gridline_count + 1):
            temp_value = self.min_temp_value + i * (self.max_temp_value - self.min_temp_value) / self.gridline_count
            label = self.font.render(f"{temp_value:.0f}", True, DARK_GRAY)
            self.screen.blit(label, (self.x - 40, self.y + self.height - i * self.height / self.gridline_count - label.get_height() / 2))
        title = self.font.render("Temperature Over Time", True, BLACK)
        self.screen.blit(title, (self.x + (self.width - title.get_width()) / 2, self.y - 30))

class ControlPanel:
    def __init__(self, screen, env, width, height ,x ,y):
        self.screen = screen
        self.env = env
        self.font = pygame.font.Font('freesansbold.ttf', 24)

        margin = width // 10
        self.meter_width = width // 2 - margin
        self.meter_max_height = height

        self.y = y
        self.x = x
        
        self.heater_meter_x = x
        self.cooler_meter_x = x + self.meter_width + margin

    def update(self):
        # Draw heating and cooling meters
        heating_meter_height = int(self.env.heating_meter * self.meter_max_height / self.env.max_meter)
        cooling_meter_height = int(self.env.cooling_meter * self.meter_max_height / self.env.max_meter)

        pygame.draw.rect(self.screen, RED, (self.heater_meter_x, self.y - heating_meter_height, self.meter_width, heating_meter_height))
        pygame.draw.rect(self.screen, BLUE, (self.cooler_meter_x, self.y - cooling_meter_height, self.meter_width, cooling_meter_height))

        # Display text for meters
        heat_text = self.font.render(f"H {self.env.heating_meter:.0f}", True, BLACK)
        self.screen.blit(heat_text, (self.heater_meter_x, self.y + 10))

        cool_text = self.font.render(f"C {self.env.cooling_meter:.0f}", True, BLACK)
        self.screen.blit(cool_text, (self.cooler_meter_x, self.y + 10))

        # Display current temperature
        current_temp = self.env.current_temperature
        temp_text = self.font.render(f"Temp: {current_temp:.1f}°C", True, BLACK)
        
        self.screen.blit(temp_text, (self.x, self.y + 30))

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                self.env.step(0)  # Heat up
            elif event.key == pygame.K_c:
                self.env.step(1)  # Cool down

# Initialize Pygame
pygame.init()
env = SmartHomeTempControlEnv()
state = env.reset()

# Set up the display
width, height = 1200, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('SmartTemp-RL')

# Create instances of Graph and ControlPanel
panel_width, panel_height = 150, 200
graph = Graph(screen, env, width // 2 , height // 2 , 100, 50)
control_panel = ControlPanel(screen, env, panel_width, panel_height , width // 2 - panel_width // 2, height // 2 + panel_height + 100)

# Main loop
running = True
last_action_time = time.time()
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        else:
            control_panel.handle_event(event)

    # Automatic step
    if time.time() - last_action_time > 0.5:
        env.step(2)  # Automatic 'S' action
        last_action_time = time.time()

    # Fill screen with white
    screen.fill(WHITE)

    # Update and draw graph
    current_temp = env.current_temperature
    graph.update(current_temp)

    # Update and draw control panel
    control_panel.update()

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    pygame.time.Clock().tick(60)

# Close Pygame and exit
pygame.quit()
sys.exit()
