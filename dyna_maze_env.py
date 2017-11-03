#!/usr/bin/env python


# Avery Tan altan 1392212
# CMPUT366 A5


"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np


head_probability = 0.55 # head_probability: floating point
num_total_states = 99 # num_total_states: integer
obstructions = [(2,1),(2,2),(2,3),(5,4),(7,0),(7,1),(7,2)]


currX=None
currY=None
goalX=None
goalY=None

def env_init():
    """
    initializes the board and starting positions
    state is the board represented by a list of list whereby curr_state[0][5] 
    is the board position in the first 1st and 6th column, (6,1).
    """

def env_start():
    """ returns a tuple representing the starting position of the agent"""
    global currX,currY,goalX,goalY


    #initialize start and goal states
    currX=0
    currY=2
    goalX=8
    goalY=0

    curr_coor=(currX,currY)
    return curr_coor

def env_step(action):
    """
    Arguments
    ---------
    action : tuple containing (x,y) where x and y are integers where x~[0,1] and y~[0,1]

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
        reward : float
        state: tuple representing (x,y) where x and y are integers
        isTerminal : bool
    """
    global current_state,currX,currY,goalX,goalY


    oldX=currX
    oldY=currY

    is_terminal = False

    #the agent will not select an action if it has found the Terminal state or when the condition for termination is reaached
    #in the case of n-step sarsa 
    if action == None:
        is_terminal=True
    else: #the agent has chosen an action, calculate the next state
        actionX=action[0]
        actionY=action[1]

        currX+=actionX
        currY+=actionY

    #check if we have tried to go to a barrier/obstructions tile
    #if so, set our curr coordinates to the previous coordinates
    if (currX,currY) in obstructions:
        currX=oldX
        currY=oldY

    #if the agent tries to move out of the boundary of the gridworld
    if currY>5:
        currY=5
    elif currY<0:
        currY=0
    if currX>8:
        currX=8
    elif currX<0:
        currX=0

    
    #determine reward
    reward = 0.0
    if currX==goalX and currY==goalY:
        reward+=1
        is_terminal=True
    else:
        reward=0

    curr_coor=(currX,currY)
    

    result = {"reward": reward, "state": curr_coor, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
