import random

class Trainer:
    def __init__(self, states=None, state_transitions=None, state=None):

        self.states = states

        self.state_transitions = state_transitions
    
    def train(self, m, n=1):
        for i in range(0, n):
            s = Sim(self.states.copy(), self.state_transitions.copy(), self.states.copy()[0]) # (..., self.state or self.initial_state)
            m.train(s) # or s.train(m)

class Real:
    def __init__(self, states=None, state_transitions=None, state=None):
        self.states = states
        self.state_transitions = state_transitions
        self.state = state
    def alter(self, action):
        if action in self.state_transitions[self.state]:
            print("real world moves from " + self.state, end="")
            self.state = self.state_transitions[self.state][action]
            print(" to " + self.state)
        else:
            self.state = self.state # unchanged, the NULL action
        

class Sim:
    def __init__(self, states=None, state_transitions=None, state=None):
        self.states = set()
        self.state_transitions = state_transitions # of 
        self.state = state
        
    def alter(self, action):
        if action in self.state_transitions[self.state]:
            print("'world' moves from " + self.state, end="")
            self.state = self.state_transitions[self.state][action]
            print(" to " + self.state)
        else:
            self.state = self.state # unchanged, the NULL action    
    


class Model:
    def __init__(self, observations=None, actions=None, weights=None, rewards=None):
        self.observations = observations
        self.actions = actions
        self.weights = dict()
        self.rewards = rewards
    
    
    def action(self, observation):
        a = random.choice(actions)
        return a
    
    def observe(self, state):
        return self.observations[state]
    
    def reward(self, state):
        return rewards[state]
        
    def train(self, sim):
        for j in range(0, 100):
            observation = self.observe(sim.state)
            action = self.action(observation)
            sim.alter(action)
            reward = self.reward(self.observe(sim.state))
            if observation in self.weights.keys():
                if action in self.weights[observation].keys():
                    self.weights[observation][action] = reward
                else:
                    self.weights[observation][action] = reward
            else:
                self.weights[observation] = dict()
                self.weights[observation][action] = reward
    
    def execute(self, real):
        for j in range(0, 100):
            observation = self.observe(real.state)
            action = self.policy[observation]
            real.alter(action)
            
                
    def freeze(self):
        policy = dict()
        for observation in self.weights.keys():
            policy[observation] = max(self.weights[observation], key=self.weights[observation].get)
        self.policy = policy

    
        
observations = {"a1":"A1", "a2":"A2", "b1":"B1","b2":"B2"}

states = ["a1", "a2", "b1", "b2"]

actions = ["X", "Y"]

state_transitions = {"a1":{"X":"a2","Y":"b1"}, "a2":{"X":"a2", "Y":"b2"}, "b1":{"X":"b2","Y":"b1"}, "b2":{"X":"b2", "Y":"b2"}}

rewards = {"A1":0.1, "B1":-0.5, "A2":0.5, "B2":1}

def trainer():
    t = Trainer(states, state_transitions, states[0])
    return t

def model():
    m = Model(observations, actions, rewards)
    return m

def real():
    r = Real(states, state_transitions, states[0])
    return r   

if __name__ == "__main__":
    import sys
    m = Model(observations, actions, rewards)
    t = Trainer(states, state_transitions, states[0])
    t.train(m, int(sys.argv[1]))
    m.freeze()
    r = Real(states, state_transitions, states[0])
    m.execute(r)
    
    
    
