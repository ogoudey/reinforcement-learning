import random

class Trainer:
    def __init__(self):

        self.states = states

        self.state_transitions = state_transitions
    
    def train(self, m, n=1):
        for i in range(0, n):
        
            s = Sim(self.states.copy(), self.state_transitions.copy(), self.states.copy()[0]) # (..., self.state or self.initial_state)
            m.train(s) # or s.train(m) 

class Sim:
    def __init__(self, alter_function, initial_state, state_transitions, rewards):
        self.alter = alter_function
        self.initial_state = initial_state
        self.state = initial_state
        self.state_transitions = state_transitions
        self.rewards = rewards
        
    def reset(self):
        self.state =  self.initial_state
           
    def test(self, policy, n=100, h=100):
        for i in range(0, n):
            self.reset()
            rewards = 0
            for j in range(0, h):
                action = policy.action(self.state)
                self.state, reward = self.alter(self.state, action)
                rewards += reward
                
                # run until finish
                # but which states are available when?
                # lets make G -> G so only G reward for each episode
        


class Model:
    def __init__(self, policy):
        self.returns = dict()
        self.policies = [policy]
        self.rewards = None
        
    def step(self, sim):
        sim.alter(sim.state, action)
        

    def iterate_with(self, sim):
        state = self.initial_state
        for j in range(0, self.horizon):
            action = self.pick_action(state)
            
            self.step(action, sim)

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
        if self.random:
            action = random.choice(list(self.actions))
            return action
        if state in self.action_selections.keys():
            action = self.action_selections[state]
        else:
            action = None
        return action
    
    def from_QTable_greedy(self, table):
        # Assuming table - policy/action agreement
        for state in states:
            self.action_selections[state] = max(table.table[state], key=table.table[state].get)
        
"""
class State:
    def __init__(self):
        self.content = ""

    def value(self, policy, sim):
        if not sim:
            sim = Sim(alter, initial_state, state_transitions, rewards)
        i = 1
        _return = 0
        while i < 5:
            action = policy.action(state)
            __, reward = sim.alter(action)
            _return += reward * (gamma ** i) # discounting
            i += 1
        return _return
"""

def value(state, action, policy, sim=None):
    if not sim:
        sim = Sim(alter, initial_state, state_transitions, rewards)
    i = 0
    _return = 0
    state_p = state
    rewards = []
    while i < 5:
        action = policy.action(state_p)
        state_p, reward = sim.alter(state_p, action)
        _return += reward * (gamma ** i) # discounting
        rewards.append(reward * (gamma ** i))
        i += 1
    return _return    

class QTable:
    def __init__(self, states):
        self.states = states
        self.table = dict()
        pass

    def train2(self, base_policy, n, sim):
        for state in states:
            average_return = 0
            for i in range(0, n):
                action = base_policy.action(state)
                average_return += value(state, action, base_policy, sim)
            average_return /= n
            if not state in self.table.keys():
                self.table[state] = dict()
            self.table[state][action] = average_return
    
    def train(self, base_policy, n, sim):
        for state in states:
            average_returns = dict()
            action_cnts = dict()
            for i in range(0, n):
                action = base_policy.action(state)
                if not action in average_returns.keys():
                    average_returns[action] = 0
                    action_cnts[action] = 0
                average_returns[action] += value(state, action, base_policy, sim)
                action_cnts[action] += 1
            for action in average_returns:
                average_returns[action] /= action_cnts[action]
            
            if not state in self.table.keys():
                self.table[state] = dict()
            for action in average_returns.keys():
                self.table[state][action] = average_returns[action]
                print(state + " with action " + action + " has " + str(average_returns[action]))
            print(action_cnts)
# "environment" variables
"""
states = set()
actions = set()
state_transitions = {states:{actions:states}} # S x A x S -- currently not python
rewards = {}
initial_state = "..."
"""               

gamma = .5

states = {"S", "1", "2", "3", "4", "G"}
actions = {"N", "S", "E", "W"}
policy_choices = {"S":"N", "2":"N", "1":"E", "3":"E", "4":"E"}
state_transitions = {"S":{"N":"2", "S":"S"}, "2":{"N":"1"}, "1":{"E":"3", "S":"S"}, "3":{"E":"4"}, "4":{"E":"G"}, "G":{"E":"G"}} # S x A x S -- currently not python
rewards = {"4":{"E":100}, "1":{"S":10}}
initial_state = "1"

def alter(state, action):
    if action:
        if state in state_transitions.keys():
            if action in state_transitions[state]: # should be if Line l - 5 is complete
                new_state = state_transitions[state][action]
            else:
                new_state = state
        else:
            new_state = state
            
        if state in rewards.keys():
            if action in rewards[state]:
                reward = rewards[state][action]
            else:
                reward = 0
        else:
            reward = 0
            
    else:
        new_state = state
        reward = 0
        
    return new_state, reward
    

# should take a policy, an attribute of a model
def run(model, sim, episode_length=10):
    state = initial_state
    for step in range(0, episode_length):
        action = random.choice(list(model.policies[0].actions))
        state, __ = sim.alter(state, action)
        # why we doing this?

def teleop():
    s = Sim(alter, initial_state, state_transitions, rewards)
    state = s.initial_state
    while True:
        action = input(">")
        state, reward = s.alter(state, action)
        print(state + ", " + str(reward))
        # Could collect data in this time, for human_policy
        
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
s = Sim(alter, initial_state, state_transitions, rewards)
policy = Policy(states, actions, preset=None, random=True)
s.test(policy)
qtable = QTable(states)
qtable.train(policy, 36, s) # Qtable now has returns from random walks from all states
new_policy = Policy(states, actions, preset=None, random=False)
new_policy.from_QTable_greedy(qtable.table) # policy now has valuable actions
s.test(new_policy)


if __name__ == "__main__":
    policy = Policy(states, actions, policy_choices)
    m = Model(policy)
    num_iterations = 1
    #m.iterate_with(sim)
    for i in range(0, num_iterations):
        sim = Sim(alter, initial_state, state_transitions, rewards)
        run(m, sim)
        
"""
Add two policies, random and greedy.
"""

    
    
    
