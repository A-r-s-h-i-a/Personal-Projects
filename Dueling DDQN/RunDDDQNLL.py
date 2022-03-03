from DDDQNKeras import Agent
import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("darkgrid")

if __name__ == '__main__':
	env = gym.make('LunarLander-v2')
	n_games = 500
	agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[8], epsilon_dec=1e-3, mem_size=100000, batch_size=64, 
												n_actions=4, eps_end=0.01, fc1_dims=128, fc2_dims=128, replace=100)
	
	scores, eps_history = [], []

	for i in range(n_games):
		done = False
		score = 0
		observation = env.reset()
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward
			agent.store_transition(observation, action, reward, observation_, done)
			observation = observation_
			agent.learn()
		eps_history.append(agent.epsilon)
		scores.append(score)

		avg_score = np.mean(scores[-100:])
		print('epsisode ', i, 'score %.1f' % score,
				'average score %.1f' % avg_score,
				'epsilon %.2f' % agent.epsilon)

	filename = 'keras_lunar_lander.png'
	generation = [i+1 for i in range(n_games)]
	data_plot = pd.DataFrame({"Generation":generation, "Score":scores})
	sns.relplot(x='Generation', y='Score', data=data_plot, kind="scatter")
	plt.show()