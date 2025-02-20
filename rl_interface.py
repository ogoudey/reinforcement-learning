import rl

def main():

    rp = rl.Policy(rl.states, rl.actions, None, True)
    
    s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)
    
    new_policy = rl.monte_carlo(rp, initial_state=(10,10), episode_length=100, horizon=100, cycles=1)
    
    return s, new_policy
    
    



