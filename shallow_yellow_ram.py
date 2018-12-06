#Uses Deep Q-Learning to approximate the optimal policy for decision making
#during the game, where the end goal is to maximize the game score
#Breakout

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import plot_model
from collections import deque
import gym
import time as t
import numpy as np

total_episodes = 5

class shallow_yellow:
    def __init__(self, input_image_size, available_actions):
        self.state_size = input_image_size
        self.action_size = available_actions
        self.memories = deque(maxlen = 10000) #How long should the AI hold onto memories? Once the deque is full, it begins forgetting old memories from when the AI sucked
        self.epsilon = 0.95 #Initial Exploration rate, as a percent probability that the chosen action will be random
        self.epsilon_decay = 0.9 #How quickly the AI stops exploring and starts to trust its decsions
        self.epsilon_minimum = 0.025 #The lowest possible exploration rate
        self.learning_rate = 0.001 #The size of each step of gradient descent
        self.model = self.compile_network()

    def compile_network(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer="adam", loss="mse", learning_rate=self.learning_rate)
        return model

    def print_network_structure_graph(self):
        plot_model(self.model, to_file='shallow_yellow_ram_structure.png')

    def remember(self, s_t,a_t,r_t,s_t1): #Uses Q-Learning Notation
        self.memories.append(s_t,a_t,r_t,s_t1)



if __name__ == "__main__": #This is structured around the Q-Learning Algorithim
    env = gym.make('Breakout-ram-v0') #Creates an instance of Atari Breakout
    state_size = env.observation_space.shape[0] #Asks the environment how big the screen input will be #For some reason is a tuple, so first value is retrieved
    action_size = env.action_space.n #Asks the environment how many possible action can be taken
    SY = shallow_yellow(state_size, action_size)

    for episode in range(total_episodes):
        current_state = env.reset()
        total_reward = 0
        for t in range(5000):
            env.render() #Remove for quicker training, allows you to watch the game be played
            random_action = env.action_space.sample()
            next_state, reward, game_over, debug_info = env.step(random_action)
            if game_over:
                print("Episode {} finished after executing {} actions".format(episode, t+1))
                break
            ##Save transition to memory
            current_state = next_state

    SY.print_network_structure_graph()
    print("Training Complete.")
