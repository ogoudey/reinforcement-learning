import rl

def main():
    rp = rl.Policy(rl.states, rl.actions, None, True)
    
    s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)
    
    avg_rets = rl.monte_carlo(rp, initial_state=(10,10), episode_length=100, horizon=100, rollouts=1000)
    
    return avg_rets
    
    



