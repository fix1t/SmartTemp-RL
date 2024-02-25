import sys
import torch
import time
from agent.agent import Agent
from smart_home_env import SmartHomeTempControlEnv

# Expected usage: python agent_replay.py <filename>
if len(sys.argv) != 2:
    print("Usage: python agent_replay.py <filename>")
    sys.exit(1)

env = SmartHomeTempControlEnv()
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
agent = Agent(state_size, number_actions)

filename = sys.argv[1]
#TOOD: Verify if the file exists...

# Load the model
checkpoint = torch.load(filename)
agent.local_qnetwork.load_state_dict(checkpoint)

state = env.reset(False)
env.render()
total_reward = 0
done = False

time_step = 0
max_time_steps = 4*24*180

while not done or time_step < max_time_steps:
    action = agent.act(state, epsilon=0)  # Using epsilon=0 for evaluation
    state, reward, done, _ = env.step(action)
    total_reward += reward
    # print("Action:", action, "Reward:", reward)
    time.sleep(0.001)
    time_step += 1
print("Total reward:", total_reward)
