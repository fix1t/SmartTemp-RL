from smart_home_env import SmartHomeTempControlEnv

env = SmartHomeTempControlEnv()
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
