import rl

def main():


    s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)
    
    m = Model(1000)

    
    


    #s.visualize(p, 100, (0,0)) # stuck in (0,0)-(1,0) loop
    #s.visualize(p, 100, (1,1)) #same
    #s.visualize(p, 100, (2,2)) #same
    #s.visualize(p, 100, (3,3))
    #s.visualize(p, 100, (4,4))
    rl.teleop(m.policies)

    
class Model:
    def __init__(self, rollouts=1000, horizon=100):
        rp = rl.Policy(rl.states, rl.actions, None, True)

        qt = rl.QTable(rl.states)

        s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)

        qt.train(rp, rollouts, s, horizon)

        self.primary_policy = rl.Policy(rl.states, rl.actions, None, True)

        self.primary_policy.from_QTable_greedy(qt, 0)
        
        self.secondary_policy = rl.Policy(rl.states, rl.actions, None, True)

        self.secondary_policy.from_QTable_greedy(qt, 1)
        
        self.policies = [self.primary_policy, self.secondary_policy]
