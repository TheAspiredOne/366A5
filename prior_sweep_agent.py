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


pi = None
last_action=None
alpha = 0.1
epsilon = 0.1
currX=None
currY=None
Q=None
gamma = 0.95
model = None
old_state = None
random.seed(a=1000)
N=5
theta = 0.00066
maxP = None
PQueue=None

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
    global Q,model, prev_obs_states,prev_obs_actions_in_states,pi,PQueue
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
    PQueue = dict()



def record_StateAndAction(os,la):
    '''
    function adds a state and action to 'memory'
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
    returns the action with the largest estimated curr Q value as a tuple (x,y)
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
    global currX,currY,Q, epsilon,alpha,last_action,old_state,gamma,n,N,pi,theta,PQueue
    
    # simple worldmap implementation to visualize what is happening
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


    #update the model
    model[(old_state,last_action)]=(reward,state)

    action = get_max_a_Q(currX,currY)
    #LEARN!
    Q[old_state[0]][old_state[1]][last_action]=Q[old_state[0]][old_state[1]][last_action]+alpha*(reward+gamma*Q[currX][currY][action]-Q[old_state[0]][old_state[1]][last_action])


    #Calculate the priority P
    max_a=get_max_a_Q(currX,currY)
    priority=abs(reward+gamma*Q[currX][currY][max_a]-Q[old_state[0]][old_state[1]][last_action])
    if priority>theta:
        #ensure that if a state action pair already is in the queue, that this new priority is greater.
        if (old_state,last_action) in PQueue:
            if priority > PQueue[(old_state,last_action)]:
                PQueue[(old_state,last_action)]=priority
        else:
            PQueue[(old_state,last_action)]=priority
    
    #doing the planning update:
    if N != 0:
        n=random.randint(1,N)
        count_n=0
        while count_n<n and PQueue:
            count_n+=1
            S,A = max(PQueue, key=PQueue.get)
            maxP=PQueue[max(PQueue, key=PQueue.get)]
            del PQueue[(S,A)] #pop

            re,next_state = model[(S,A)]
            caction = get_max_a_Q(next_state[0],next_state[1])
            Q[S[0]][S[1]][A]+=alpha*(re+gamma*Q[next_state[0]][next_state[1]][caction]-Q[S[0]][S[1]][A])
            r_sp=None
            for i in model:
                if model[i][1]==S: #for every state leading to this state
                    r_sp = model[i][0]
                    preceding_state = i[0]
                    preceding_action = i[1]
                    max_a=get_max_a_Q(S[0],S[1])
                    priority = abs(r_sp+gamma*Q[S[0]][S[1]][max_a]-Q[preceding_state[0]][preceding_state[1]][preceding_action])
                    if priority > theta:
                        if (preceding_state,preceding_action) in PQueue:
                            if priority > PQueue[(preceding_state,preceding_action)]:
                                PQueue[(preceding_state,preceding_action)]=priority
                        else:
                            PQueue[(preceding_state,preceding_action)]=priority
        
    #now action selection for the next direct RL
    rndm=rand_un()
    if rndm < epsilon: #explore!
        poss_moves= [(1,0),(-1,0),(0,1),(0,-1)]
        action_index=rand_in_range(len(poss_moves))
        action = poss_moves[action_index]
    
    else: #exploit!    
        action = get_max_a_Q(currX,currY)
        pass

    old_state = (currX,currY)
    last_action=action
    #adding to list of seen state
    record_StateAndAction(old_state,last_action)
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global Q,old_state,last_action,alpha,N

    model[(old_state,last_action)]=(reward,(8,0))

    #LEARN!
    Q[old_state[0]][old_state[1]][last_action]=Q[old_state[0]][old_state[1]][last_action]+alpha*(reward-Q[old_state[0]][old_state[1]][last_action])


    #Calculate the priority P
    max_a=get_max_a_Q(currX,currY)
    priority=abs(reward-Q[old_state[0]][old_state[1]][last_action])
    if priority>theta:
        if (old_state,last_action) in PQueue:
            if priority > PQueue[(old_state,last_action)]:
                PQueue[(old_state,last_action)]=priority
        else:
            PQueue[(old_state,last_action)]=priority

  
    #doing the planning update:
    if N != 0:
        n=random.randint(1,N)
        count_n=0
        while count_n<n and PQueue:
            count_n+=1
            S,A = max(PQueue, key=PQueue.get)
            maxP=PQueue[max(PQueue, key=PQueue.get)]
            del PQueue[(S,A)]

            re,next_state = model[(S,A)]
            caction = get_max_a_Q(next_state[0],next_state[1])
            Q[S[0]][S[1]][A]+=alpha*(re+gamma*Q[next_state[0]][next_state[1]][caction]-Q[S[0]][S[1]][A])
            r_sp=None
            for i in model:
                if model[i][1]==S:
                    r_sp = model[i][0]
                    preceding_state = i[0]
                    preceding_action = i[1]
                    max_a=get_max_a_Q(S[0],S[1])
                    priority = abs(r_sp+gamma*Q[S[0]][S[1]][max_a]-Q[preceding_state[0]][preceding_state[1]][preceding_action])
                    if priority > theta:
                        if (preceding_state,preceding_action) in PQueue:
                            if priority > PQueue[(preceding_state,preceding_action)]:
                                PQueue[(preceding_state,preceding_action)]=priority
                        else:
                            PQueue[(preceding_state,preceding_action)]=priority
    
    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
        
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"

