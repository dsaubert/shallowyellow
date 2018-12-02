#Daniel Saubert - CS495 Senior Project
#Shallow Yellow is a reinforcement based machine learning program that uses
#OpenAI Gym to interface with environments

import tensorflow as tf
import gym
import numpy as np

env = gym.make('SpaceInvaders-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()
