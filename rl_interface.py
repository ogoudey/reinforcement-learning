import rl

def main():

    rp = rl.Policy(rl.states, rl.actions, None, True)
    s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)
    
    #m = Model(1000)
    qt = rl.QTable(rl.states)
    qt.train(rp, 1000, s, 100)

    p = rl.Policy(rl.states, rl.actions)

    p.from_QTable_greedy(qt)

    s.visualize(p, 100, (5,5))
    
    
    m = Model()
    rl.teleop(m.policies)
    print("Returning (sim, policy)")
    return s, p


    
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
