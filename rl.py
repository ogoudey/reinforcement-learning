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
        self.state = initial_state
        self.state_transitions = state_transitions
        self.rewards = rewards
        
        # currently holds no actual state
        
        


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
    def __init__(self, states, actions, preset=None):
        if preset:
            self.action_selections = preset # States -> A
        self.states = states
        self.actions = actions
    
    def action(self, state):
        
        if state in self.action_selections.keys():
            action = self.action_selections[state]
        else:
            action = None
        return action

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


def value(state, policy, sim=None):
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
    print(rewards)
    return _return    


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
state_transitions = {"S":{"N":"2"}, "2":{"N":"1"}, "1":{"E":"3", "S":"S"}, "3":{"E":"4"}, "4":{"E":"G"}} # S x A x S -- currently not python
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
    


def run(model, sim, episode_length=10):
    state = initial_state
    for step in range(0, episode_length):
        action = random.choice(list(model.policies[0].actions))
        state, __ = sim.alter(state, action)
        # why we doing this?

if __name__ == "__main__":
    policy = Policy(states, actions, policy_choices)
    m = Model(policy)
    num_iterations = 1
    #m.iterate_with(sim)
    for i in range(0, num_iterations):
        sim = Sim(alter, initial_state, state_transitions, rewards)
        run(m, sim)

    
    
    
