#!/usr/bin/env python


# Avery Tan altan 1392212
# CMPUT366 A5


"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle
import operator
import random

alpha_values = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0] #our alpha values to be used
alpha_counter = 0


alpha_state = 1
pi = None
last_action=None
alpha = 0.1
epsilon = 0.05
currX=None
currY=None
Q=None
gamma = 0.95
model = None
old_state = None
random.seed(a=1000)
N=5
random_flag = 0

prev_obs_states= None
prev_obs_actions_in_states= None


def calculate_poss_moves(xx,yy):
    """
    calculates possible moves given current x and y coordinates
    returns a list containing all valid legal moves
    """

    global currX , currY
    action_space=[(1,0),(-1,0),(0,1),(0,-1)]
    legal_moves=list()

    #filtering out illegal moves:
    for i in action_space:
        if i[0]+currX>=0 and i[0]+currX<=8 and i[1]+currY>=0 and i[1]+currY<=5:
            legal_moves.append(i)

    return legal_moves




def agent_init():
    """
    Hint: Initializes Q, where we store the Q(s,a). Q is a 2d array such that Q[x][y][a] where x and y are
    intergers and represent a particular location and a is an action and Q[x][y][a] is the state-action value for
    that particular action taken in that particular state
    Returns: nothing
    """
    global Q,model, prev_obs_states,prev_obs_actions_in_states,pi
    prev_obs_states=set()
    prev_obs_actions_in_states=dict()
    Q=list()
    for i in range(9):
        Q.append([{(1,0):0,(-1,0):0,(0,1):0,(0,-1):0},
            {(1,0):0,(-1,0):0,(0,1):0,(0,-1):0},
            {(1,0):0,(-1,0):0,(0,1):0,(0,-1):0},
            {(1,0):0,(-1,0):0,(0,1):0,(0,-1):0},
            {(1,0):0,(-1,0):0,(0,1):0,(0,-1):0},
            {(1,0):0,(-1,0):0,(0,1):0,(0,-1):0}])
    model=dict()
    random.seed(a=1000)
    pi=dict()



def record_StateAndAction(os,la):
    '''
    put the old state and previous action to 'memory'
    '''
    global prev_obs_states,prev_obs_actions_in_states
    #adding to list of seen state
    prev_obs_states.add(os)
    if os not in prev_obs_actions_in_states:
        prev_obs_actions_in_states[os]=list()
        prev_obs_actions_in_states[os].append(la)
    else:
        prev_obs_actions_in_states[os].append(la)

    return



def get_max_a_Q(currX,currY):
    '''
    returns the action with the highest estimated Q
    '''
    global Q
    poss_moves=[(1,0),(-1,0),(0,-1),(0,1)]
    qlist=dict()
    for i in poss_moves:
        qlist[i]=Q[currX][currY][i]
    max_q=qlist[max(qlist, key=qlist.get)]

    #choose randomly all optimal legal moves
    equal_q_list=list()
    for i in poss_moves:
        if Q[currX][currY][i] >= max_q:
            equal_q_list.append(i)
    action_index=rand_in_range(len(equal_q_list))
    action = equal_q_list[action_index]

    return action




def agent_start(state):
    """
    Arguments: state: tuple (x,y) representing coordinates on gridworld
    Returns: action: tuple (x,y) representing an action
    """
    global currX,currY,Q,last_action,model,pi, old_state

    currX=state[0]
    currY=state[1]


    
    equal_q_list=list()
    action_space = [(1,0),(-1,0),(0,1),(0,-1)]


    #getting the highest q
    qlist=dict()
    for i in action_space:
        qlist[i]=Q[currX][currY][i]
    max_q=qlist[max(qlist, key=qlist.get)]

    
    #going through possible actions again to get all estimated optimal actions
    for i in action_space:
        if qlist[i]==max_q:
            equal_q_list.append(i)


    #randomly choose an action
    action_index=rand_in_range(len(equal_q_list))
    action = equal_q_list[action_index]

    old_state = (currX,currY)

    #adding to list of seen state
    record_StateAndAction(old_state,action)

    last_action=action
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: tuple (x,y) representing coordinates of agent
    Returns: action: tuple (x,y) representing move agent can make
    """
    global random_flag,currX,currY,Q, epsilon,alpha,last_action,old_state,gamma,n,N,pi,alpha_counter,alpha_values
    
    if alpha_state==2: #determine whether performing alpha sweep exp.
    	alpha = alpha_values[alpha_counter]

    #simple wolrd map
    # worldmap = list()
    # obstructions = [(2,1),(2,2),(2,3),(5,4),(7,0),(7,1),(7,2)]
    # for u in range(9):
    #     worldmap.append(['.','.','.','.','.','.'])
    # worldmap[8][0]='G'
    # worldmap[currX][currY]='s'
    # for j in obstructions:
    #     worldmap[j[0]][j[1]]='*'
    # for i in worldmap:
    #     print i
    # print currX,currY
    # print '\n\n'


    action = None
    currX=state[0]
    currY=state[1]


    #must find the action which produces max Q in order to perform our direct RL update
    action = get_max_a_Q(currX,currY)
    #LEARN!
    Q[old_state[0]][old_state[1]][last_action]=Q[old_state[0]][old_state[1]][last_action]+alpha*(reward+gamma*Q[currX][currY][action]-Q[old_state[0]][old_state[1]][last_action])

    #update the model
    model[(old_state,last_action)]=(reward,state)

    #doing the planning update:
    if N != 0:
	    n=random.randint(1,N)
	    for i in range(n):
	        rand_state = random.sample(prev_obs_states,1)[0]
	        len_potential_action = len(prev_obs_actions_in_states[rand_state])
	        rand_action = prev_obs_actions_in_states[rand_state][rand_in_range(len_potential_action)]
	        re,next_state = model[(rand_state,rand_action)]

	        #must select max Q(s,a)
	        max_a = get_max_a_Q(next_state[0],next_state[1])
	        Q[rand_state[0]][rand_state[1]][rand_action]=Q[rand_state[0]][rand_state[1]][rand_action]+alpha*(re+gamma*Q[next_state[0]][next_state[1]][max_a]-Q[rand_state[0]][rand_state[1]][rand_action])
	    
    #now action selection for the next direct RL
    rndm=rand_un()
    if rndm < epsilon: #explore!
        poss_moves= [(1,0),(-1,0),(0,1),(0,-1)]
        action_index=rand_in_range(len(poss_moves))
        action = poss_moves[action_index]
    
    else: #exploit!    
        # action = get_max_a_Q(currX,currY)
        pass

    pi[(currX,currY)] =action
    old_state = (currX,currY)
    last_action=action
    #adding to list of seen state
    record_StateAndAction(old_state,action)
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global Q,old_state,last_action,alpha,N
    if alpha_state==2:
    	alpha = alpha_values[alpha_counter]

    Q[old_state[0]][old_state[1]][last_action]=Q[old_state[0]][old_state[1]][last_action]+alpha*(reward-Q[old_state[0]][old_state[1]][last_action])
    model[(old_state,last_action)]=(reward,(8,0))
    #doing the planning update:
    if N!=0:
	    n=random.randint(1,N)
	    for i in range(n):
	        rand_state = random.sample(prev_obs_states,1)[0]
	        len_potential_action = len(prev_obs_actions_in_states[rand_state])
	        rand_action = prev_obs_actions_in_states[rand_state][rand_in_range(len_potential_action)]
	        re,next_state = model[(rand_state,rand_action)]

	        #must select max Q(s,a)
	        max_a = get_max_a_Q(next_state[0],next_state[1])
	        Q[rand_state[0]][rand_state[1]][rand_action]=Q[rand_state[0]][rand_state[1]][rand_action]+alpha*(re+gamma*Q[next_state[0]][next_state[1]][max_a]-Q[rand_state[0]][rand_state[1]][rand_action])


    
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
          
        
    return

def agent_message(in_message): # returns string, in_message: string
    global Q,alpha_counter,alpha_state
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    elif(in_message == 'UpdateAlpha'): #exp has told us to mvoe to next alpha value
    	alpha_counter+=1
    elif (in_message=='UpdateAlphaStatus'): #exp has told us we are doing alpha sweep exp.
    	alpha_state = 2
    else:
        return "I don't know what to return!!"

