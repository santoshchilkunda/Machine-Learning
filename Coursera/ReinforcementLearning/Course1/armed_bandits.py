#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

class kArmedBandits():

	def __init__(self, k, problems, steps, epsilon):
		self.k = k
		self.problems = problems
		self.epsilon = epsilon
		self.steps = steps

		# problems x k
		self.action_value_map = self._create_action_value_map_(problems, k)

		# problems x steps
		self.epsilon_map = self._create_epsilon_map_(epsilon, problems, steps)

		# problems x steps
		self.rewards = np.zeros((problems, steps), dtype=np.float32)

		# steps x 1
		self.rewards_norm = np.zeros(steps, dtype=np.float32)

	def run(self):
		self._run_tests_()
		self._normalize_reward_()

		return self.rewards_norm[:self.steps]

	# private methods

	# problems x k
	# gaussian distribution with zero mean and unit variance
	def _create_action_value_map_(self, problems, k):
		action_value_map = np.zeros((problems,k), dtype=np.float32)

		# (TBD) vectorize?
		for i in range(problems):
			action_value_map[i,:] = np.random.normal(0.0, 1.0, size=(k))

		return action_value_map

	# problems x steps
	# uniform distribution between (0,1)
	# threshold epsilon to create a binary map
	def _create_epsilon_map_(self, epsilon, problems, steps):
		epsilon_map = np.zeros((problems, steps), dtype=np.float32)

		# (TBD) vectorize?
		for i in range(problems):
			epsilon_map[i,:] = np.random.uniform(0.0, 1.0, size=(steps))

		epsilon_map = epsilon_map > epsilon

		return epsilon_map

	def _run_tests_(self):
		# (TBD) vectorize
		for t in range(self.steps):
			exploit = self.epsilon_map[:,t]
			for p in range(self.problems):
				act_val_p = self.action_value_map[p,:]

				if exploit[p]:
					_k = np.argmax(act_val_p)
				else:
					_k = np.random.randint(0, (self.k-1))

				act_val = act_val_p[_k]

				# reward: gaussian distribution with mean q(a) and var 1
				self.rewards[p, t] = np.random.normal(act_val, 1.0)

	def _normalize_reward_(self):
		# normalize accross problems axis
		self.rewards_norm = np.mean(self.rewards, axis=0)

# norm_rewards_arr: num_epsilons x steps
def plot_results(norm_rewards_dict):
	x_len = list(norm_rewards_dict.values())[0].shape[0]
	x_axis = np.linspace(0, x_len, x_len)

	fig = plt.figure()
	for eps,rewards in norm_rewards_dict.items():
		plt.plot(x_axis, rewards, label=str(eps))

	plt.title('K-Armed Bandits', fontsize=14)
	plt.xlabel('Steps', fontsize=14)
	plt.ylabel('Average Reward', fontsize=14)
	plt.legend(loc='upper right')
	plt.grid(True)

	fig.savefig('kab_results.png')

# Entry point
if __name__ == '__main__':

	k = 50
	problems = 4000
	steps = 1000
	epsilon_list = [0, 0.01, 0.1]

	norm_reward_dict = OrderedDict()
	for eps in epsilon_list:
		print('Running k-armed bandits for epsilon %s' % eps)
		k_armed_bandits = kArmedBandits(k, problems, steps, eps)
		norm_reward_dict[eps] = k_armed_bandits.run()

	plot_results(norm_reward_dict)

	print('Complete')
