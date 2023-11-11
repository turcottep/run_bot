import gym
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
    # action = env.action_space.sample()

    # get all possible actions
    actions = env.action_space.n

    print("actions", actions)

    action = 0

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    env.render()

env.close()

print("Done")
