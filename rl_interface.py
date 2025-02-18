import rl

def main():
    rp = rl.Policy(rl.states, rl.actions, None, True)

    qt = rl.QTable(rl.states)

    s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)

    qt.train(rp, 10000, s, 100)

    p = rl.Policy(rl.states, rl.actions)

    p.from_QTable_greedy(qt)


    #s.visualize(p, 100, (0,0)) # stuck in (0,0)-(1,0) loop
    #s.visualize(p, 100, (1,1)) #same
    #s.visualize(p, 100, (2,2)) #same
    #s.visualize(p, 100, (3,3))
    #s.visualize(p, 100, (4,4))
    s.visualize(p, 100, (5,5))
    print("Returning (sim, policy)")
    return s, p

    
class TrainedPolicy:
    def __init__(self):
        rp = rl.Policy(rl.states, rl.actions, None, True)

        qt = rl.QTable(rl.states)

        s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)

        qt.train(rp, 1000, s, 100)

        self.policy = rl.Policy(rl.states, rl.actions)

        self.policy.from_QTable_greedy(qt)
