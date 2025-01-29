from tic_tac_toe.env import TicTacToeEnv
import random
import pickle

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        self.q_table = {}  # Stores state-action values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay over time

    def load_q_table(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

        print("Q-table loaded from", path)

    def get_q_value(self, state, action):
        """Returns Q-value for state-action pair, defaulting to 0"""
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, env: TicTacToeEnv, state, action, reward, next_state):
        """Updates Q-value using the Q-learning formula"""
        max_future_q = max([self.get_q_value(next_state, a) for a in env.available_actions()], default=0)
        self.q_table[(state, action)] = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (reward + self.gamma * max_future_q)

    def choose_action(self, env: TicTacToeEnv):
        state = env.get_state()

        # Check for winning and blocking moves first
        for check_move in [env.check_win_move, env.check_block_move]:
            move = check_move(env.current_player)
            if move:
                return move

        available_actions = env.available_actions()

        # Decide to explore or exploit
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)  # Explore: random action

        # Exploit: choose the best action based on the Q-table
        best_action = max(available_actions, key=lambda action: self.q_table.get((state, action), 0))
        return best_action
    
    def decay_epsilon(self):
        """Reduces exploration over time"""
        self.epsilon *= self.epsilon_decay
