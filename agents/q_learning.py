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

    def choose_action(self, state, available_actions, game_env):
        # First, check if the AI has a winning move
        winning_move = game_env.check_win_move(self.current_player)
        if winning_move:
            return winning_move, 1  # Reward for making a winning move

        # Then, check if the AI needs to block the opponent
        blocking_move = game_env.check_block_move(self.current_player)
        if blocking_move:
            return blocking_move, 0.5  # Reward for blocking the opponent

        # If no win or block, choose randomly or based on the Q-table
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions), 0  # Explore
        else:
            # Exploit: choose the best action based on the Q-table
            state_q_values = {action: self.q_table.get((state, action), 0) for action in available_actions}
            best_action = max(state_q_values, key=state_q_values.get)
            return best_action, 0  # No immediate win/block, just the best known action
    
    def decay_epsilon(self):
        """Reduces exploration over time"""
        self.epsilon *= self.epsilon_decay
