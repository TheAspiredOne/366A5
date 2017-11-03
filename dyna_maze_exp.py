#!/usr/bin/env python


# Avery Tan, altan, 1392212
# CMPUT366 A4


"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

"""


from rl_glue import * 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
	#see if the user input a particular experiment to run, if not, default is sarsa(0)
	exp_to_run = 'dyna_q_agent'
	if len(sys.argv)>=2:
		exp_options = sys.argv[1]
		if exp_options == 'n':
			exp_to_run = 'prior_sweep_agent'
		else:
			exp_to_run = 'dyna_q_agent'

	print '\nrunning ', exp_to_run,'experiment...'
	#initialize RL_glue
	RLGlue("dyna_maze_env", exp_to_run)
	num_episodes = 50
	max_steps = 100000
	num_runs = 10
	step_array = np.zeros(num_episodes+1)
	steps_per_runs = np.zeros(num_episodes+1)
	
	for jk in range(num_runs):
		RL_init()
		# run the experiment for each episode
		for episode in range(num_episodes):
			print 'Episode ',episode,' begins,..'
			RL_episode(max_steps)
			num_steps = RL_num_steps()
			step_array[episode+1] =  num_steps
		RL_cleanup()
		steps_per_runs += step_array
	steps_per_runs /= num_runs




np.save(exp_to_run,steps_per_runs)