import gymnasium as gym

# Initialise the environment
env = gym.make(
    "LunarLander-v3",
    # render_mode="human",
    continuous=True,
)

# Show the environment info
print("# action_space")
print("dims: ", env.action_space.shape[0])
print("high: ", env.action_space.high)
print("low : ", env.action_space.low)

print("# observation_space")
print("dims: ", env.observation_space.shape[0])
print("high: ", env.observation_space.high)
print("low : ", env.observation_space.low)


# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(100):
    done = False
    total_reward = 0.0

    while True:
        action = env.action_space.sample()
        action = [0.00, 0.4]
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            print("total_reward: ", total_reward)
            break

env.close()

# Action space
# bottom engine : action[0]
# side engines : action[1]
#   - if action[1] > 0, then the left engine is on
#   - if action[1] < 0, then the right engine is on

# Observation space
# lander pos x : observation[0]
# lander pos y : observation[1]
# lander vel x : observation[2]
# lander vel y : observation[3]
# lander angle : observation[4]
# lander angular velocity : observation[5] (increased when the lander rotates counter-clockwise)
# lander right leg contact : observation[6]
# lander left leg contact : observation[7]
