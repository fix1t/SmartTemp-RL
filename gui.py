import pygame
import sys
from smart_home_env import SmartHomeTempControlEnv

# Initializing the Gym environment
env = SmartHomeTempControlEnv()
state = env.reset()

pygame.init()

screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption('SmartTemp-RL')

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

font = pygame.font.Font(None, 36)

running = True
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                state, _, _, _ = env.step(0)  # Heat up
            elif event.key == pygame.K_c:
                state, _, _, _ = env.step(1)  # Cool down
            elif event.key == pygame.K_s:
                state, _, _, _ = env.step(2)  # Stop HVAC

    screen.fill(WHITE)

    # Display the current temperature
    temp_text = font.render(f"Temp: {state[0]:.2f}°C", True, BLACK)
    screen.blit(temp_text, (20, 20))

    # Display the heating meter
    heat_text = font.render(f"Heating: {env.heating_meter:.2f}", True, RED)
    screen.blit(heat_text, (20, 60))

    # Display the cooling meter
    cool_text = font.render(f"Cooling: {env.cooling_meter:.2f}", True, BLUE)
    screen.blit(cool_text, (20, 100))

    # Update the display
    pygame.display.flip()

# Close the Pygame window
pygame.quit()
sys.exit()
