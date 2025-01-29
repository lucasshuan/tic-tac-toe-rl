from agents.q_learning import QLearningAgent
from tic_tac_toe.env import TicTacToeEnv
import pickle

env = TicTacToeEnv()
agent = QLearningAgent()

num_episodes = 10000  # Number of training games

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state, env.available_actions())
        next_state, reward, done = env.step(action)
        agent.update_q_value(env, state, action, reward, next_state)
        state = next_state

    agent.decay_epsilon()

save_path = "models/q_table.pkl"

with open(save_path, "wb") as f:
    pickle.dump(agent.q_table, f)

print("Training complete!")
print("Q-table saved to", save_path)