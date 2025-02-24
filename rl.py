import random
import copy
import time

from tqdm import tqdm

import numpy as np

np.set_printoptions(threshold=np.inf)
states = set()
state_transitions = dict()
m = np.zeros((12,12))
mapped = []

def _map(y,x):
    states.add((y,x))
    mapped.append((y,x))
    state_transitions[(y,x)] = dict()
    if x + 1 >= m.shape[0]:
        state_transitions[(y,x)]["E"] = (y,x)
    else:
        state_transitions[(y,x)]["E"] = (y,x + 1)
        if not (y,x + 1) in mapped:
            _map(y,x+1)
    if x - 1 < 0:
        state_transitions[(y,x)]["W"] = (y,x)
    else:
        state_transitions[(y,x)]["W"] = (y,x - 1)
        if not (y,x - 1) in mapped:
            _map(y,x-1)
    if y + 1 >= m.shape[0]:
        state_transitions[(y,x)]["S"] = (y,x)
    else:
        state_transitions[(y,x)]["S"] = (y + 1,x)
        if not (y+1,x) in mapped:
            _map(y+1,x)
    if y - 1 < 0:
        state_transitions[(y,x)]["N"] = (y,x)
    else:
        state_transitions[(y,x)]["N"] = (y-1,x)
        if not (y-1,x) in mapped:
            _map(y-1,x)

_map(0,0)
state_transitions[(11,11)] = {"N":(11,11), "S":(11,11), "E":(11,11), "W":(11,11)}
gamma = 0.9
actions = {"N", "S", "E", "W"}
rewards = {(11,10):{"E":100}, (10,11):{"S":100}}
initial_state = (0,0)

def value(state, action, policy, sim, horizon=5):
    sim.reset()
    state_p, reward = sim.alter(state, action)
    _return = reward
    returns = []
    #returns.append(str(reward) + " * "+str(gamma) + " ** " +str(0)+ " = " + str(reward * (gamma ** 0)))
    i = 1
    while i < horizon:
        action = policy.action(state_p)
        state_p, reward = sim.alter(state_p, action)

        _return += reward * (gamma ** i) # discounting
        #returns.append(str(reward) + " * "+str(gamma) + " ** " +str(i)+ " = " + str(reward * (gamma ** i)))
        i += 1
    #print(returns)
    return _return    

def render(state, reward=0):
    n = m.copy()
    n[state] = 1
    print(n)
    print(str(state) + ", " + str(reward))

class Sim:
    def __init__(self, initial_state, state_transitions, init_rewards={(11,10):{"E":100}, (10,11):{"S":100}}):
        self.initial_state = initial_state
        self.state = initial_state
        self.state_transitions = state_transitions
        self.init_rewards = init_rewards # preserve init rewards
        self.rewards = copy.deepcopy(self.init_rewards)
        
    def reset(self, initial_state=None):
        if initial_state:
            self.state = initial_state
        else:
            self.state =  self.initial_state
        self.rewards = copy.deepcopy(self.init_rewards)
        
    def visualize(self, policy, h, initial_state=None, no_reset=False):
        if not initial_state:
            initial_state = self.initial_state
        if no_reset:
            pass
        else:
            self.reset(initial_state)
        render(self.state)
        time.sleep(1.5)
        try:
            for j in range(0, h):
               
               action = policy.action(self.state)
               self.state, reward = self.alter(self.state, action)
               render(self.state, reward)
               if reward == 100:
                   render(self.state, "100 reward received.")
                   time.sleep(3)
                   break
               time.sleep(1)
        except KeyboardInterrupt:
            return self.state
            print("\nExiting visualization.")
           
           
           
           
    def test(self, policy, n=1000, h=100, initial_state=None):
        rewards = 0
        for i in range(0, n):
            self.reset(initial_state)
            for j in range(0, h):
                action = policy.action(self.state)
                self.state, reward = self.alter(self.state, action)
                rewards += reward
        rewards /= n

        print("Average cumulative reward over " + str(n) + " tries: " + str(rewards))

        
    def alter(self, state, action):
        if action:
            if state in self.state_transitions.keys():
                if action in self.state_transitions[state]: # should be if Line l - 5 is complete
                    new_state = self.state_transitions[state][action]
                else:
                    new_state = state
            else:
                new_state = state
                
            if state in self.rewards.keys():
                if action in self.rewards[state]:
                    reward = self.rewards[state][action]
                    #print("resetting " + str(self.rewards[state][action]))
                    self.rewards[state][action] = 0 # So rewards can not be 'farmed'
                    #print(self.init_rewards[state][action])
                else:
                    reward = 0
            else:
                reward = 0
                
        else:
            new_state = state
            reward = 0
            
        return new_state, reward

class Policy:
    def __init__(self, states, actions, preset=None, random=False):
        if preset:
            self.action_selections = preset # States -> A
        else:
            self.action_selections = dict()
        self.states = states
        self.actions = actions
        self.random = random
    
    def action(self, state):
        
        if state in self.action_selections.keys():
            action = self.action_selections[state]
        else:
            if self.random:
                action = random.choice(list(self.actions))
            else:
                action = None
        return action
    
    def from_table_greedy(self, table):
        # Assuming table - policy/action agreement
        print("My policy's states: " + str(len(self.action_selections.keys())) + "\nState overlap: " + str(len(list(table.keys()))/len(list(self.states))))
        
        for state in tqdm(self.states):
            if state in table.keys():
            
                self.action_selections[state] = max(table[state], key=table[state].get)
            else:
                #print("State " + str(state) + " has no data in table.")
                pass


class QTable:
    def __init__(self, states):
        self.states = states
        self.table = dict()
        pass

    
    def train(self, base_policy, n, sim, horizon):
        self.table = dict() # restart table
        for state in tqdm(self.states):
            average_returns = dict()
            action_cnts = dict()
            for i in range(0, n):
                action = base_policy.action(state)
                if not action in average_returns.keys():
                    average_returns[action] = 0
                    action_cnts[action] = 0
                average_returns[action] += value(state, action, base_policy, sim, horizon)
                action_cnts[action] += 1
            for action in average_returns.keys():
                average_returns[action] /= action_cnts[action]
            
            if not state in self.table.keys():
                self.table[state] = dict()
            for action in average_returns.keys():
                self.table[state][action] = average_returns[action]

# MC v1
def rollout_monte_carlo(init_policy, initial_state=None, episode_length=100, horizon=100, rollouts=100):
    average_returns = dict()
    action_pair_cnts = dict()
    s = Sim(initial_state, state_transitions)
    policy = init_policy
    for r in range(0, rollouts): # It's not really rollouts but improvement cycles...
        if not initial_state:
            initial_state = random.choice(states)
        s.reset(initial_state)  
        state = s.state  
        states = []
        actions = []
        rewards = []
        for step in range(0, episode_length):
            states.append(state)
            if not state in average_returns.keys():
                average_returns[state] = dict()
                action_pair_cnts[state] = dict()
            action = policy.action(state)
            actions.append(action)
            state, reward = s.alter(state, action)
            rewards.append(reward)
        #print(states)
        #print(actions)
        #print(rewards)
        for i in range(0, episode_length):
            _return = 0
            j = 0
            while (j + i) < len(rewards):
                _return += (gamma**j) * rewards[j + i]
                j += 1
            if actions[i] in average_returns[states[i]].keys():
                average_returns[states[i]][actions[i]] += _return
                action_pair_cnts[states[i]][actions[i]] += 1
            else:
                average_returns[states[i]][actions[i]] = _return
                action_pair_cnts[states[i]][actions[i]] = 1
    #print(average_returns)
    #print(action_pair_cnts)
    for state in average_returns.keys():
        for action in average_returns[state].keys():    
            average_returns[state][action] /= action_pair_cnts[state][action]
        
    #print(average_returns)
    
    return average_returns

    
def monte_carlo(init_policy, initial_state=None, episode_length=100, horizon=100, cycles=100):
    average_returns = dict()
    action_pair_cnts = dict()
    s = Sim(initial_state, state_transitions)
    policy = init_policy
    for c in range(0, cycles): # It's not really rollouts but improvement cycles...
        if not initial_state:
            initial_state = random.choice(states)
        s.reset(initial_state)  
        state = s.state  
        states = []
        actions = []
        rewards = []
        for step in range(0, episode_length):
            states.append(state)
            if not state in average_returns.keys():
                average_returns[state] = dict()
                action_pair_cnts[state] = dict()
            action = policy.action(state)
            actions.append(action)
            state, reward = s.alter(state, action)
            rewards.append(reward)
        #print(states)
        #print(actions)
        #print(rewards)
        for i in range(0, episode_length):
            _return = 0
            j = 0
            while (j + i) < len(rewards):
                _return += (gamma**j) * rewards[j + i]
                j += 1
            if actions[i] in average_returns[states[i]].keys():
                average_returns[states[i]][actions[i]] += _return
                action_pair_cnts[states[i]][actions[i]] += 1
            else:
                average_returns[states[i]][actions[i]] = _return
                action_pair_cnts[states[i]][actions[i]] = 1
        #print(average_returns)
        #print(action_pair_cnts)
        for state in average_returns.keys():
            for action in average_returns[state].keys():    
                average_returns[state][action] /= action_pair_cnts[state][action]
        policy.from_table_greedy(average_returns)
    
    print(average_returns)
    return policy

        

def teleop(fallback_policy=None):
    gen_policy = dict()
    s = Sim(initial_state, state_transitions, rewards.copy())
    state = s.initial_state
    render(state, "...")
    try:    
        while True:
            action = input("> ")
            if action == "":
                if not fallback_policy: # last minute
                    fallback_policy = Policy(states, actions, None, True)
                state = s.visualize(fallback_policy, 100, state)
            else:
                if not state in gen_policy.keys():
                    gen_policy[state] = dict()
                if not action in gen_policy[state].keys():
                    gen_policy[state][action] = 0
                gen_policy[state][action] += 1
                state, reward = s.alter(state, action)
            render(state, reward)
            # Could collect data in this time, for human_policy
    except KeyboardInterrupt:
            
            action_selections = dict()
            for state in gen_policy.keys():
                action_selections[state] = max(gen_policy[state], key=gen_policy[state].get)
            h_policy = Policy(states, actions, action_selections, True) 
            print("\nGenerated policy.")
            return h_policy
# for all states:
# if I do action a in state s (then act randomly) what's the expected reward
# gets VVVVV
# "Q for s = "0" and a = "0" is number
#
# Yeah to learn a model, a policy, just sample all state-action pairs.
# make a printer for state-action pairs
# with first sim, get a Q-function; then use those values for a new policy - test that policy.
# Turn the dictionary into a matrix?
#
# So the MDP has transition probabilities... theres a policy
# now improve MDP
#
#
# Then, states are bins of continuous space 


