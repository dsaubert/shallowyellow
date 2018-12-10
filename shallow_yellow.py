#Uses Deep Q-Learning to approximate the optimal policy for decision making
#during the game, where the end goal is to maximize the game score
#Breakout

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape
from keras import optimizers
from keras.utils import plot_model
from collections import deque
import gym
import time as t
import numpy as np
from scipy import misc
import random
from skimage import color
import gc

total_episodes = 10000
memory_replay_sample_size = 32
minimum_replay_size = 50000
replay_frequency = 4 #How many steps to perform before learning again.
save_model_frequency = 100 #Episodes in between saving iteration

class shallow_yellow:
    def __init__(self, input_image_size, available_actions):
        self.state_size = input_image_size
        self.action_size = available_actions
        self.memories = deque(maxlen = 100000) #How long should the AI hold onto memories? Once the deque is full, it begins forgetting old memories from when the AI sucked
        self.epsilon = 0.99 #Initial Exploration rate, as a percent probability that the chosen action will be random
        self.epsilon_decay = 0.99 #How quickly the AI stops exploring and starts to trust its decsions
        self.epsilon_minimum = 0.01 #The lowest possible exploration rate
        self.learning_rate = 0.001 #The size of each step of gradient descent
        self.discount_rate = 0.95 #Ratio of future action reward to current action reward. Ensures success in the long run while prioritizing realtime success
        self.model = self.compile_network()

    def compile_network(self): #As defined in the 2013 Nature paper
        model = Sequential()
        model.add(Conv2D(32, 8, strides=(4,4), padding="valid", activation="relu", input_shape=(4,84,80), batch_size=1, data_format="channels_first")) #The format of the data to be supplied is [i, j, k, l] where i is the number of training samples, j is the height of the image, k is the weight and l is the channel number.
        model.add(Conv2D(64, 4, strides=(2,2), padding="valid", activation="relu"))
        model.add(Conv2D(64, 1, strides=(3,3), padding="valid", activation="relu"))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer="adam", loss="mse") #default adam LR is 0.001, mse is defined in Q-eqn of (optimal - guessed)^2
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
            self.model.fit(current_state, q_function_target, batch_size=memory_replay_sample_size, epochs=1, verbose=0) #train the model to accurately compute the actual Q value, and don't tell me you're training.

    def update_epsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay

    #Takes a new frame, returns the processed version.
    def pre_process_new_frame(self, a_frame):
         return self.rgb2gray(self.downscale(a_frame))

    #Crops out score and ceiling from frame, as well as a few pixels below
    #the paddle
    def downscale(self, big_frame):
        return big_frame[36:-7:2,::2,:]

    #Reduces the RGB pixel value into a single greyscale luminosity value
    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [.3, .6, .1])

    def save(self):
        self.model.save(filepath="ShallowYellowModelProperPreProcessing.h5",overwrite=True)
        print("Saving Model")
        return

### Main Functions ###

#Oh, the wonderful laws of matrix modification. Adding the none parameter is crucial
#for the new frame to be included, so that the dimensions line up
def add_frame_to_state(a_state, a_frame):
    new_state = a_state[1:]
    return np.concatenate([new_state, a_frame[None,...]], axis=0)

#Plays a single game (episode) of Breakout
def play_episode(env, SY, game_over, current_state):
    for t in range(50000):

        #Allows the user to watch the NN play
        env.render()

        #Ask the NN for an action, which may be random, depending on the
        #current epsilon value it has
        action = SY.choose_action(current_state)

        #Perform the chosen action, and capture the response from the environment
        next_frame, current_reward, game_over, debug_info = env.step(action)

        #Each frame needs to be pre-processed, to save on computational complexity.
        #This includes downsampling the entire image
        next_state = add_frame_to_state(current_state, SY.pre_process_new_frame(next_frame))
        SY.remember(current_state, action, current_reward, game_over, next_state)
        if game_over:
            return(t)
        current_state = next_state
        if ((len(SY.memories) > minimum_replay_size) & (t % replay_frequency == 0)): #if there are enough memories available, perform experience replay
            SY.perform_memory_replay(memory_replay_sample_size)
        SY.update_epsilon()

if __name__ == "__main__":
    #Creates an instance of Atari Breakout that automatically skips 4 frames
    env = gym.make('BreakoutDeterministic-v4')

    #Defines an instance of a Keras neural network, whose output is limited
    #to the number of availale actions within the environment
    state_size = env.observation_space.shape
    action_size = env.action_space.n #Asks the environment how many possible action can be taken
    SY = shallow_yellow(state_size, action_size)

    #Loops through episodes, each of which is an entire 3-Life game of Atari Breakout
    for episode in range(total_episodes):

        #Begins the game, and receives the start screen.
        initial_frame = env.reset()
        processed_initial_frame = SY.pre_process_new_frame(initial_frame)
        #To 'prime' the initial state, it needs 4 previous memories,
        #which are all set to the initial frame.
        first_state = np.array([processed_initial_frame, processed_initial_frame, processed_initial_frame, processed_initial_frame])

        #Play the current episode, which returns the ending amount of actions
        #it was able to execute
        time_steps_completed = play_episode(env, SY, False, first_state)

        print("Episode {} finished after executing {} actions.".format(episode, time_steps_completed))

        #Periodically save the model
        if episode % save_model_frequency == 0:
            SY.save()

    SY.print_network_structure_graph()
    print("Training Complete.")
