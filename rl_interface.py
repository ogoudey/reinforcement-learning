import rl

def main():

    rp = rl.Policy(rl.states, rl.actions, None, True)
    
    s = rl.Sim(rl.initial_state, rl.state_transitions, rl.rewards)
    p = rl.Policy(rl.states, rl.actions, None, True)
    table = rl.rollout_monte_carlo(rp, initial_state=(0, 0), episode_length=100, horizon=100, rollouts=10000)
    
    # table = rl.rollout_monte_carlo(rp, initial_state=(0, 0), episode_length=100, horizon=100, rollouts=10000)
    
    p.from_table_greedy(table)
    
    return s, p
