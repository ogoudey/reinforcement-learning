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
        self.tick = 0
        
    def alter(self, action):
        if action in self.state_transitions[self.state]:
            print("'world' moves from " + self.state, end="")
            self.state = self.state_transitions[self.state][action]

            print(" to " + self.state)
        else:
            self.state = self.state # unchanged, the NULL action    
        self.tick += 1    


class Model:
    def __init__(self, observations=None, actions=None, weights=None, rewards=None, epsilon=1, horizon=100):
        
        self.actions = actions

        self.weights = dict()

        self.rewards = rewards
        self.horizon = horizon
        
        self.observer = observations
        self.prior_observations = ["A1", "A2", "B1", "B2"] # Let the observations change over time, let new replace old, let nothing be completely new, and nothing completely deja vuet
    
    def action(self, observation):
        if random.random() < epsilon:
            a = random.choice(actions)
        else:
            a = max(self.weights[observation_name], key=self.weights[observation_name].get)
        return a
    """ Obselete:
    def observe(self, state):
        return self.observations[state]
    """
    def reward(self, state):
        return rewards[state]
        
    def train(self, sim):
        for j in range(0, self.horizon):
            observation = Observation(sim.state, j, self.observer, self.prior_observations)
            action = self.action(observation.content)
            sim.alter(action)
            reward = self.reward(Observation(sim.state, self.observer, self.prior_observations).content)
            if observation.content in self.weights.keys():
                if action in self.weights[observation.content].keys():
                    self.weights[observation.content][action] = reward
                else:
                    self.weights[observation.content][action] = reward
            else:
                self.weights[observation.content] = dict()
                self.weights[observation.content][action] = reward
    
    def execute(self, real):
        for j in range(0, 100):
            observation = Observation(real.state, self.observer, self.prior_observations)
            action = self.policy[observation.content]
            real.alter(action)
            
                
    def freeze(self):
        policy = dict()
        for observation_content in self.weights.keys():
            policy[observation_content] = max(self.weights[observation_name], key=self.weights[observation_name].get)
        self.policy = policy
        return policy
    
class Observation:
    def __init__(self, trace, time, observer=None, priors=None): # trace ought to be minimal, observer is a substitute function for 'make'
        self.object = trace
        self.time = time
        self.observer = observer
        self.priors = priors
        
        self.content = self.make(trace)
        self.time
        
    def make(self, trace):
        for name in self.priors:
            # cheaty similarity function
            if trace.capitalize() == name:
                # Add "new" observation,
                return name #
        
class Agent:
    def __init__(self, policy=None):
        self.master_policy = policy
        self.frozen_policies = dict()
        self.tasks = ["chug"]
        
    def execute(self, plan):
        for action in plan:
            if action in self.tasks:
                print(action)
            else:
                # rl
                environment = e # another perceived thing
                observation = "A1"
                # What's the environment? target?
                if environment in environments:
                    if theres a model for environment,
                        does it converge?
                        if not 
                if not create environment
                    make a model
                    converge it
            all of this happens on seperate threads
                
                
                
                
                  
    def recall(self):
        if env in self.frozen_policies.keys():
            policy = self.frozen_policies[env]
        else:
            # make env, train, act
            pass
            

    
        
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
    print("Executed with policy " + str(m.policy))
    
    
    
