import socket
import os
import sys
from PIL import Image
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import numpy as np
import random
import time

from collections import deque

from keras.callbacks import TensorBoard
from tensorflow.summary import FileWriter
import tensorflow


OBSERVATION_SPACE_SIZE = (1,28)
ACTION_SPACE_SIZE = 2
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 500  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 4  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 100

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


# Own Tensorboard class, used to ignore lots of the operations done per call to fit()
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        #Main Model - used to actually fit
        self.model = self.create_model()
        
        #Target model - used to predict, updated every so episodes or epochs
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) #Used to create 'batches' for fitting
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0 #Tracking how many more examples to see before updating target_model
        
        
        
    def create_model(self):
        #Create model for generating Q values
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=1, input_shape=OBSERVATION_SPACE_SIZE))
        model.add(Activation("relu"))
        model.add(MaxPooling1D(1))
        model.add(Dropout(0.2))
        
        model.add(Conv1D(256, 1))
        model.add(Activation("relu"))
        model.add(MaxPooling1D(1))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(64))
        
        model.add(Dense(ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr = 0.001), metrics=["accuracy"])
        
        return model
    
    
    def update_replay_memory(self, transition):
        #Update replay_memory with new (state, action, reward, new_state, done)
        self.replay_memory.append(transition)
        
        
    def get_qs(self, terminal_state):
        return self.model.predict(np.array(terminal_state).reshape(-1, *terminal_state.shape)/255)[0]
    
    def train(self, terminal_state):
        #Grab a sample from replay_memory, use as batch to fit() target_model

        print("Training")
        
        #If replay_memory is too small, sampling from it will always return same sample and will result in overfitting
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        #Get Q Values
        current_states = np.array([transition[0] for transition in minibatch])/255 #Normalizing and sample states
        current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done: #Perform operations; environment not over
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
                
            current_qs = current_qs_list[index]
            
            current_qs[action] = new_q
            
            X.append(current_state)
            y.append(current_qs)
            
        self.model.fit(np.array(X)/255, np.array(y), batch_size = MINIBATCH_SIZE,
                      verbose = 0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        #Update count for updating target_model
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            #Update target_model
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
class JSettlersServer:

	def __init__(self, host, port, dqnagent, timeout=None):
		# Used for training agent
		self.agent = dqnagent
		self.host = host
		self.port = port
		self.timeout = timeout
		self.state = None
		self.last_action = None

		#Used for logging models and stats
		self.ep_rewards = [-200]
		self.curr_episode = 1


	def run(self):
		soc = socket.socket()         # Create a socket object
		if self.timeout:
			soc.settimeout(self.timeout)
		try:
		    soc.bind((self.host, self.port))

		except socket.error as err:
		    print('Bind failed. Error Code : ' .format(err))

		soc.listen(10)
		print("Socket Listening ... ")
		while True:
		    try:
		        conn, addr = soc.accept()     # Establish connection with client.
		        print("Got connection from",addr)
		        print("Waiting for message ... ")
		        length_of_message = int.from_bytes(conn.recv(2), byteorder='big')
		        msg = conn.recv(length_of_message).decode("UTF-8")
		        print(msg)
		        action = self.handle_msg(msg)
		        #conn.send(str(action).encode(encoding='UTF-8'))
		        print("Sent: ", action)
		    except socket.timeout:
		        print("Timeout or error occured. Exiting ... ")
		        break

	def get_action(self, state):
		if np.random.random() > epsilon:
			action = np.argmax(self.agent.get_qs(state))
		else:
			action = np.random.randint(0, ACTION_SPACE_SIZE)
		return action



	def handle_msg(self, msg):
		msg_args = msg.split("|")
		print(msg_args)
		# reward = msg_args[1]
		# if msg_args[0] == "trade": #We're still playing a game; update our agent based on the rewards returned and take an action
		# 	next_state = map(int, msg_args[2].split(","))
		# 	if self.state:	# If we have a previous state, run a train step on the agent for the last action taken
		# 		self.agent.update_replay_memory((self.state, self.last_action, reward, next_state))
		# 		self.agent.train(False)
		# 		# Update actions so that on the next step, we'll train on these actions
		# 		self.state = next_state
		# 		action = self.get_action(self.state)
		# 		self.last_action = action
		# 		return action
		# elif msg_args[0] == "end": #The game has ended, update our agent based on the rewards, update our logs, and reset for the next game
		# 	self.agent.update_replay_memory((self.state, self.last_action, reward, next_state))
		# 	self.agent.train(True)
		# 	# Update actions so that on the next step, we'll train on these actions
		# 	self.state = None
		# 	self.last_action = None
		# 	    # Append episode reward to a list and log stats (every given number of episodes)
		#     self.ep_rewards.append(reward)
		#     if not self.curr_episode % AGGREGATE_STATS_EVERY or episode == 1:
		#         average_reward = sum(self.ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
		#         min_reward = min(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
		#         max_reward = max(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
		#         self.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

		#         # Save model
		#         self.agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

		# 		# Decay epsilon
		#     if epsilon > MIN_EPSILON:
		#         epsilon *= EPSILON_DECAY
		#         epsilon = max(MIN_EPSILON, epsilon)

		#     return None



if __name__ == "__main__":
	dqnagent = DQNAgent()
	server = JSettlersServer("localhost", 2004, dqnagent, timeout=180)
	server.run()
