# NOTES ---------------------------------------------------------------------------------------------------------------
#
# DQN vs Dueling DQN Algorithm
# 	Deep Q Learning (DQN) uses a Neural Network (NN) to learn a policy, replacing "Q Tables" in the original Q Learning
# algorithm. This approach enables agents to operate in continuous environments where large amounts of input data must
# be processed for any hope of good performance. The NN takes in the agent's state (in some form) as input, and 
# produces scores for the available actions as output. If following the policy (exploiting), the agent would then take
# the action with the greatest score. The NN asks and answers the question - What action produces the greatest return?
#	Dueling DQN performs the same function as the DQN algorithm, but with four NNs and to the ends of two separate 
# questions. What is the value of the current state, and how much better is an action when COMPARED to the others? All 
# input information is passed through an initial layer before moving forwards through the system. The outputs are then
# passed to two parallel NNs, one responsible for calculating the Value of the agent's state, and one for calculating
# the Advantage of available actions. The outputs of both the Value and Advantage network, which are scalars, are then
# summed into a final layer which outputs the Q values for the system. The idea is to have learned: 
# Q = Value(s) + Advantage(s,a) 
# However, during training, the equation used for Q will be:
# Q = Value(s) + (Advantage(s,a) - Advantage(s,a).mean()) 
# 	Although this distinction loses some of the semantics, it is VERY important to subtract the mean of the Advantage
# network in the Q function as it forces the Value network to actually learn the Value function. This is because over
# many samples, the second two terms should average to zero and the single-output Value network will be effectively 
# learning as if it was the output of the entire system. Thus, over many training samples, the Advantage network outputs
# the advantage of each action for each state (but again, averages to zero when subtracted by the mean over many samples),
# the Value network outputs the value of the state, and the Q network the decision of the system for that state.
# 	The Dueling DQN algorithm provides a few major advantages over traditional DQN. Firstly, it is more efficient as in
# many scenarios it is unnecessary for an agent to estimate the action value for each action (recall in Dueling DQN only
# the advantage). Mainly, however, the advantages are during training. The fact that the Value and Advantage are 
# calculated separately in two different streams increases stability, and the Value network is updated in every update 
# of the Q values which results in a very highly learned state value. Finally, due to Advantage being the method of 
# measuring actions, the entire system/agent is more robust to noise.
#
# Algorithm's Structure
# 	There will be a class for the Agent, the Replay Memory, and the Dueling DQN. The Dueling DQN handles the estimation
# of the Action Value function. The Replay Memory keeps track of the State, Action, Reward, New State, and Terminal
# Flag transitions necessary for training the NN. The Agent is separate from both as it is not a DQN or a memory, but
# it HAS a DQN and memory which it uses to learn, make observations, choose actions, store memories, etc. Finally,
# learning will consist of the agent playing the game and using its DQN/Replay Memory granted capabilities. An epsilon
# greedy strategy will be used.
#
#---------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np



class DuelingDeepQNetwork(keras.Model):
	def __init__(self, n_actions, fc1_dims, fc2_dims):
		super(DuelingDeepQNetwork, self).__init__()
		#Input Layers
		self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
		self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
		#Value Layer
		self.V = keras.layers.Dense(1, activation=None)
		#Advantage Layer
		self.A = keras.layers.Dense(n_actions, activation=None)

	# Feedforward function
	def call(self, state):
		#Pass state through first two input layers
		x = self.dense1(state)
		x = self.dense2(x)
		#Value Stream
		V = self.V(x)
		#Advantage Stream
		A = self.A(x)
		#Output
		#Subtract the mean of the Advantage network from the Advantage Stream's output to ensure the streams properly evolve
		Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

		return Q

	# Function for extracting JUST the Advantage stream's output
	def advantage(self, state):
		x = self.dense1(state)
		x = self.dense2(x)
		A = self.A(x)

		return A

class ReplayBuffer():
	def __init__(self, max_size, input_shape):
		self.mem_size = max_size
		self.mem_cntr = 0 #variable for keeping track of first available position of memory
		#Memory Arrays
		self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	# Function for storing new state-transitions in the memory
	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_cntr % self.mem_size #position of first unoccupied memory location
		#now that we know where our first unoccupied memory location is, we just go ahead and save our new transition
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done

		self.mem_cntr+=1 #increment memory counter to avoid re-entering memories in the same location

	# Provides a batch of valid, non-repeating, non-zero memories (equal to batch_size) for training 
	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size) #identify the end of valid memories in the array (ie. are there any 0s)
													#note that mem_cntr will eventually be greater than mem_size
		batch = np.random.choice(max_mem, batch_size, replace=False) #choose from those valid memories without replacement
		#with the batch chosen, designate and return the state, new state, actions, rewards, and terminal flags
		states = self.state_memory[batch]
		new_states = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminal_memory[batch]

		return states, actions, rewards, new_states, dones

class Agent():
	def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, eps_end=0.01, mem_size=100000, 
		fname='dueling_dqn.h5', fc1_dims=128, fc2_dims=128, replace=100):
		
		# Save inputs as part of self
		self.action_space = [i for i in range(n_actions)] #list comprehension, creates list of length n_actions 
		self.gamma = gamma #for discounting future rewards
		self.epsilon = epsilon #part of epsilon-greedy strategy
		self.eps_dec = epsilon_dec
		self.eps_end = eps_end
		self.fname = fname
		self.batch_size = batch_size
		self.replace = replace #determines when we update the target network with the online eval network's weights

		# Create new parameters
		self.learn_step_counter = 0 #tracks how many times we've called the learn function
		self.memory = ReplayBuffer(mem_size, input_dims)

		# Setup Networks and init
		self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims) #create evaluation network
		self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims) #create target network for calculating cost function
		self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error') #compile model
		self.q_next.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error') #compile qnext, unnecessary as we're copying weights from q_eval

	# Interface function between agent and its memory, for replay buffer
	def store_transition(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	# Function for choosing actions based on observation of the current environment and epsilon greedy
	def choose_action(self, observation):
		# If random number is less than epsilon, do a random choice from the action space 
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.action_space)
		# Otherwise, if the random number is greater than or equal to epsilon take a greedy action
		else:
			state = np.array([observation])
			actions = self.q_eval.advantage(state)
			action = tf.math.argmax(actions, axis=1).numpy()[0]

		return action

	# Learning function
	def learn(self):
		# If the agent has NOT filled up enough of its memory equal to its batch size, then don't learn yet! Keep doing random stuff!
		if self.memory.mem_cntr < self.batch_size:
			return

		# If 100 learning function calls have occurred, update the target neural network with the online evaluation network's weights.
		if self.learn_step_counter % self.replace == 0:
			self.q_next.set_weights(self.q_eval.get_weights())

		# Feedforward
		states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size) #fetch valid memories
		q_pred = self.q_eval(states) #feed states into evaluation network
		q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy() #feed next states into target network and calc max action
		q_target = np.copy(q_pred)
				
		for idx, terminal in enumerate(dones):
			# print("\n\nQ_Target: ", q_target, "\nactions[idx]: ", actions[idx], "\nQ_Next[idx]: ", q_next[idx], "\n")
			if terminal:
				q_next[idx] = 0.0
			q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]

		self.q_eval.train_on_batch(states, q_target)

		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

		self.learn_step_counter += 1

	def save_model(self):
		self.q_eval.save(self.model_file)