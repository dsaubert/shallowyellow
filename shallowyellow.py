#Daniel Saubert - CS495 Senior Project
#Shallow Yellow is a reinforcement based machine learning program that uses
#OpenAI Gym to interface with environments
from keras.models import Sequential
from keras.layers import GRU, Input, Dense, Dropout
from keras import optimizers
import gym
import time as t
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=128))
model.add(Dense(128, activation='relu'))
model.add(Dense(18, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

episodes = 100
step_limit = 10000
env = gym.make('SpaceInvaders-v0')
avg_steps = []
env.render()


for current_episode in range (episodes):
    observation = env.reset()
    print("Episode number {}".format(current_episode))
    total_reward = 0;
    
    for step in range(step_limit):
        env.render('human')
        observation,reward,done,info= env.step(env.action_space.sample())
        total_reward += reward

        if done:
            avg_steps.append(step)
            print("Done after {} steps".format(step))
            break


print("After {} episodes, average steps/game was {}".format(episodes,np.mean(avg_steps)))
env.close()

# ACTION_MEANING = {
#     0 : "NOOP",
#     1 : "FIRE",
#     2 : "UP",
#     3 : "RIGHT",
#     4 : "LEFT",
#     5 : "DOWN",
#     6 : "UPRIGHT",
#     7 : "UPLEFT",
#     8 : "DOWNRIGHT",
#     9 : "DOWNLEFT",
#     10 : "UPFIRE",
#     11 : "RIGHTFIRE",
#     12 : "LEFTFIRE",
#     13 : "DOWNFIRE",
#     14 : "UPRIGHTFIRE",
#     15 : "UPLEFTFIRE",
#     16 : "DOWNRIGHTFIRE",
#     17 : "DOWNLEFTFIRE",
# }
