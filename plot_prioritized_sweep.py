#!/usr/bin/env python

#avery tan 1392212 altan



"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    P = np.load('prior_sweep_agent.npy')
    D = np.load('dyna_q_agent5.npy')

    plt.show()
    
    plt.plot(D, label='Dyna-Q')
    plt.plot(P, label='Prioritized Sweep')
    plt.xlim([0,50])
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title("DynaQ vs DynaQ with Prioritized Sweep")
    plt.legend()
    plt.show()