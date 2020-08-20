#!/usr/bin/env python3
import matplotlib
#matplotlib.use('Agg')

from sys import argv as sys_argv
import argparse
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

		return self.rewards_norm

	# private methods

	# problems x k
	# gaussian distribution with zero mean and unit variance
	def _create_action_value_map_(self, problems, k):
		action_value_map = np.zeros((problems,k), dtype=np.float32)

		'''
		# (TBD) vectorize?
		for i in range(problems):
			action_value_map[i,:] = np.random.normal(0.0, 1.0, size=(k))
		'''

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
		epsilon_map = np.multiply(epsilon_map, 1)

		return epsilon_map

	# problems x 1
	def _create_explore_map_(self):
		explore_map = np.zeros((self.problems), dtype=np.float32)

		# (TBD) vectorize?
		for i in range(problems):
			explore_map[i] = np.random.randint(0, self.k)

		return explore_map

	# reward: gaussian distribution with mean q(a) and var 1
	def _calc_reward_(self, mean, var=1.0):
		return np.random.normal(mean, var)

	# update action value according to:
	# Qn+1 = Qn + (1/n)*(Rn - Qn)
	# NewEstimate = OldEstimate + StepSize [ Target - OldEstimate ]
	def _update_action_value_(self, p, k, n, target):
		step_size = 1/(n+1)

		old_est = self.action_value_map[p,k]
		new_est = old_est + step_size*(target - old_est)

		self.action_value_map[p,k] = np.float32(new_est)

	def _run_tests_(self):
		p_list = np.linspace(0, self.problems, self.problems, endpoint=False)
		p_list = p_list.astype(int)

		# (TBD) vectorize
		for t in range(self.steps):
			exploit = self.epsilon_map[:,t]
			k_exploit = np.argmax(self.action_value_map, axis=1)
			k_explore = self._create_explore_map_()

			k = exploit*k_exploit + (1-exploit)*k_explore
			k = k.astype(int)

			act_val_map = self.action_value_map[p_list, k]

			for p in range(self.problems):
				act_val = act_val_map[p]

				reward = self._calc_reward_(act_val)
				self.rewards[p, t] = reward

				self._update_action_value_(p, k[p], t, reward)

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

	print('Displaying results')
	plt.show()
	#fig.savefig('kab_results.png')

# Parse arguments
def parse_args(args):
	parser = argparse.ArgumentParser(description='Script to run k-armed bandits')

	parser.add_argument('-k', '--K', type=int, required=True,
                        metavar='k',
                        help='k')

	parser.add_argument('-p', '--problems', type=int, required=True,
                        metavar='Number of problems',
                        help='Number of problems')

	parser.add_argument('-s', '--steps', type=int, required=True,
                        metavar='Number of steps',
                        help='Number of steps')

	parser.add_argument('-e', '--epsilons', type=str, required=True,
                        metavar='Comma separated list of epsilons',
                        help='Eg: 0,0.01,0.1')

	args = parser.parse_args(args)

	return args

# Entry point
if __name__ == '__main__':

	args = parse_args(sys_argv[1:])

	epsilon_list = args.epsilons.split(',')
	epsilon_list = [float(e) for e in epsilon_list]

	k = args.K
	problems = args.problems
	steps = args.steps

	norm_reward_dict = OrderedDict()
	for eps in epsilon_list:
		print('Running k-armed bandits for epsilon %s' % eps)
		k_armed_bandits = kArmedBandits(k, problems, steps, eps)
		norm_reward_dict[eps] = k_armed_bandits.run()

	plot_results(norm_reward_dict)

	print('Complete')
