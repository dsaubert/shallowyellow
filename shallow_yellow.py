#Uses Deep Q-Learning to approximate the optimal policy for decision making
#during the game, where the end goal is to maximize the game score
#Breakout

from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras import optimizers
from keras.utils import plot_model
from collections import deque
import gym
import time as t
import numpy as np
from scipy import misc
import random


total_episodes = 100
memory_replay_sample_size = 32
minimum_replay_size = 50000

class state:
    def __init__(self, state_size):
        self.size = state_size
        self.ndim = (self.size, 210, 160, 3)
        self.memory = deque(maxlen=4)

class shallow_yellow:
    def __init__(self, input_image_size, available_actions):
        self.state_size = input_image_size
        self.action_size = available_actions
        self.memories = deque(maxlen = 1000000) #How long should the AI hold onto memories? Once the deque is full, it begins forgetting old memories from when the AI sucked
        self.epsilon = 0.95 #Initial Exploration rate, as a percent probability that the chosen action will be random
        self.epsilon_decay = 0.9 #How quickly the AI stops exploring and starts to trust its decsions
        self.epsilon_minimum = 0.025 #The lowest possible exploration rate
        self.learning_rate = 0.001 #The size of each step of gradient descent
        self.discount_rate = 0.95 #Ratio of future action reward to current action reward. Ensures success in the long run while prioritizing realtime success
        self.model = self.compile_network()

    def compile_network(self):
        model = Sequential()
        model.add(Conv2D(16, (8,8), strides=4, padding="valid",input_shape=(210,160,3))) #The format of the data to be supplied is [i, j, k, l] where i is the number of training samples, j is the height of the image, k is the weight and l is the channel number.
        model.add(Conv2D(32, (4,4), strides=2, padding="valid"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def print_network_structure_graph(self):
        plot_model(self.model, to_file='\\Graphs\\shallow_yellow_ram_structure.png')

    def remember(self, s_t,a_t,r_t, done, s_t1): #Uses Q-Learning Notation
        self.memories.append((s_t,a_t,r_t,done, s_t1))

    def choose_action(self, current_state):
        explore = np.random.random() #Random [0,1)
        if explore <= self.epsilon:
            return random.randint(0, self.action_size-1)
        action_values = self.model.predict(current_state)
        return np.argmax(action_values)

    def perform_memory_replay(self, sample_size):
        memory_sample = random.sample(self.memories, sample_size)#Gathers a random sample of past choices
        for current_state, chosen_action, reward, game_over, next_state in memory_sample:
            chosen_action_target_reward = reward
            if not game_over: #then we must consider the future actions that come after this.
                chosen_action_target_reward += self.discount_rate*(np.amax(self.model.predict(next_state)))
            q_function_target = self.model.predict(current_state) #Name is not 'accurate' until next line is performed.
            q_function_target[0][action] = chosen_action_target_reward #update the reward of the chosen action, with unchanged other actions
            self.model.fit(current_state, q_function_target, epochs=1, verbose=0) #train the model to accurately compute the actual Q value, and don't tell me you're training.
        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__": #This is structured around the Q-Learning Algorithim
    env = gym.make('Breakout-v0') #Creates an instance of Atari Breakout
    state_size = env.observation_space.shape
    action_size = env.action_space.n #Asks the environment how many possible action can be taken
    SY = shallow_yellow(state_size, action_size)
    game_over = False

    def to_grayscale(img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(img):
        return img[::2, ::2]

    def preprocess(img):
        return to_grayscale(downsample(img))

    for episode in range(total_episodes):

        current_state = state(state_size)#To give the neural network a concept of trajectory and time, we train it on the last 4 frames of gameplay
        initial_state = env.reset()
        for i in range(4):
            current_state.append(initial_state)

        total_reward = 0
        print(current_state)
        for t in range(5000):
            env.render() #Remove for quicker training, allows you to watch the game be played
            action = SY.choose_action(current_state)
            next_state, reward, game_over, debug_info = env.step(action)
            next_state = current_state.copy().append(next_state) #don't want to change current state, so a copy is used.
            SY.remember(current_state, action, reward, game_over, next_state)
            if game_over:
                print("Episode {} finished after executing {} actions".format(episode, t+1))
                break
            current_state = next_state
            if (len(SY.memories) > minimum_replay_size): #if there are enough memories available, perform experience replay
                SY.perform_memory_replay(memory_replay_sample_size)

    SY.print_network_structure_graph()
    print("Training Complete.")
