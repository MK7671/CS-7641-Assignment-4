import gym
import os
import numpy as np
import time
import matplotlib.pyplot as plt
env_name  = 'Taxi-v2'
env = gym.make(env_name)
env = env.unwrapped
start=time.time()
Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []
iterations = []
optimal=[0]*env.observation_space.n
alpha = 1.0
gamma = 1.0
episodes = 20000
epsilon=0
for episode in range(episodes):
    state = env.reset()
    done = False
    t_reward = 0
    max_steps = 1000000
    for i in range(max_steps):
        if done:
            break
        current = state
        if np.random.rand()<epsilon:
            action = np.argmax(Q[current, :])
        else:
            action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
    epsilon=(1-2.71**(-episode/1000))
    alpha=2.71**(-episode/1000)
    rewards.append(t_reward)
    iterations.append(i)
for k in range(env.observation_space.n):
    optimal[k]=np.argmax(Q[k, :])
    if np.argmax(Q[k, :])==5:
        print(k)
print(optimal)
print("average :",np.average(rewards[2000:]))
env.close()
end=time.time()
print("time :",end-start)
def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = 5
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]
plt.plot(range(0, len(rewards), size)[200:], averages[200:])
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.show()
