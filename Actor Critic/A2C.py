import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt



class ActorCriticNetwork(keras.Model):
	"""
	The Actor Critic is an RL algorithm that combines Policy Gradient methods with Value-Based ones. Here,
		the policy is being approximated by a neural network (the "Actor"), and the state-action value is
		too (the "Critic"), albeit separately.
	Unlike in purely Policy Gradient methods where entire episodes must be played out by a policy before
		being updated, the addition of a Critic means that each action can be judged immediately and the
		policy updated time-step to time-step. The Value-function can likewise be updated immediately via 
		methods such as TD-lambda.
	Although the Actor and Critic are typically different networks, they can be the same one and simply
		diverge at the output. This has a couple advantages, ease of training and stability - it is what
		we will implement structurally here.
	"""
	def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512):
		super(ActorCriticNetwork, self).__init__()
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		
		#Define network layers
		self.fc1 = Dense(self.fc1_dims, activation='relu')
		self.fc2 = Dense(self.fc2_dims, activation='relu')
		self.q = Dense(1, activation=None) #State-Action Value Function
		self.pi = Dense(n_actions, activation='softmax') #Policy

	#Feedforward Function
	def call(self, state):
		value = self.fc1(state)
		value = self.fc2(value)

		#State-Action Value Function FF
		q_val = self.q(value)

		#Policy FF
		pi_val = self.pi(value)

		return q_val, pi_val



class Agent(object):
	"""
	An agent utilizing the Actor Critic method must perform a specific algorithmic loop to find the
		optimal state-action value function and optimal policy for an environment:

	1. Select the action using the Actor network
	2. Take the action and recieve the reward and new state
	3. Calculate Delta, the advantage (how much better the new state is) plus the reward
	4. Use Delta to update the parameters of the Actor and Critic networks

	Something to note, A2C is not a very robust and/or efficient algorithm. Thus, it requires large
		networks, many time-steps to train, and can quickly fall apart.
	"""
	def __init__(self, alpha, gamma, n_actions):
		super(Agent, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.n_actions = n_actions
		self.action = None
		self.action_space = [i for i in range(self.n_actions)]

		#Instantiate agent's AC networks
		self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
		self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

	def choose_action(self, observation):
		state = tf.convert_to_tensor([observation])
		_, probs = self.actor_critic(state) #Feedforward the state, get estimated state-value and policy output

		#Convert the output of the AC to a categorical distribution, ie. a representative distribution that
		#	sums to 1. Then sample from this distribution to choose an action.
		action_probabilities = tfp.distributions.Categorical(probs=probs)
		action = action_probabilities.sample()

		self.action = action

		return action.numpy()[0]

	def learn(self, state, reward, state_, done):
		state = tf.convert_to_tensor([state], dtype=tf.float32)
		state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
		reward = tf.convert_to_tensor(reward, dtype=tf.float32)

		#The Gradient Tape is a TF API for "automatic differentiation; that is, computing the gradient of a 
		#	computation with respect to some inputs". We will now use it to compute the gradient.
		with tf.GradientTape() as tape:
			#Pass the state and next state through the AC
			state_value, probs = self.actor_critic(state)
			state_value_, _ = self.actor_critic(state_)
			state_value = tf.squeeze(state_value)
			state_value_ = tf.squeeze(state_value_)

			#Convert AC outputs to probability distributions
			action_probs = tfp.distributions.Categorical(probs=probs)
			log_prob = action_probs.log_prob(self.action)

			#Calculate Temporal Difference and resulting A/C cost functions
			delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
			actor_loss = -log_prob*delta
			critic_loss = delta**2

			total_loss = actor_loss + critic_loss

		gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
		self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))



if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	agent = Agent(alpha=0.00003, gamma=0.99, n_actions=env.action_space.n)
	n_games = 2000

	filename = 'cartpole1.png'

	best_score = env.reward_range[0]
	score_history = []
	avg_history = []

	for i in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward
			agent.learn(observation, reward, observation_, done)
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		avg_history.append(avg_score)

		# #Decay alpha
		# if i%250==0 and agent.alpha>1e-5:
		# 	agent.alpha = agent.alpha/2
		# 	print("\n", agent.alpha, "\n")

		print("\nGame", i+1, " 	Score", score, "      	Average Score", avg_score)

		if avg_score > best_score:
			best_score = avg_score

	x = [i+1 for i in range(n_games)]
	plt.plot(x, avg_history)
	plt.show()