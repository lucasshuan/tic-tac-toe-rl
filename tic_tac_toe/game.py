from agents.q_learning import QLearningAgent
from tic_tac_toe.env import TicTacToeEnv
import random
import time

def get_random_move(env: TicTacToeEnv):
    """Returns a random available move."""
    available_moves = env.available_actions()
    return random.choice(available_moves)

def get_ai_move(env: TicTacToeEnv, agent: QLearningAgent):
    """Chooses an action using the agent's policy."""
    return agent.choose_action(env.get_state(), env.available_actions())

def get_player_move():
    """Prompts the player to enter their move."""
    x = int(input("Enter x (0-2): "))
    y = int(input("Enter y (0-2): "))
    return x, y

def play_tic_tac_toe():
    game = TicTacToeEnv()
    turn_count = 1
    player_token = TicTacToeEnv.X if random.randint(0, 1) == 0 else TicTacToeEnv.O
    player_starts = player_token == TicTacToeEnv.X
    
    agent = QLearningAgent()

    agent.load_q_table("models/q_table.pkl")
    
    print("Player goes first." if player_starts else "AI goes first.")

    if not player_starts:
        time.sleep(1)
        ai_move = get_ai_move(game)
        _, _, done = game.step(ai_move)  # AI makes the first move
        game.render()

    while True:
        print("Your turn.")
        player_x, player_y = get_player_move()
        try:
            _, reward, done = game.step((player_x, player_y))  # Player's move
            if done:
                game.render()
                if reward == 1:
                    print("Player wins!")
                elif reward == -1:
                    print("AI wins!")
                else:
                    print("Draw!")
                break
        except ValueError:
            print("Invalid move, try again.")
            continue

        game.render()

        time.sleep(1)

        print("AI's turn.")
        ai_move = get_ai_move(game, agent)
        _, reward, done = game.step(ai_move)  # AI's move
        
        if done:
            game.render()
            if reward == 1:
                print("Player wins!")
            elif reward == -1:
                print("AI wins!")
            else:
                print("Draw!")
            break

        game.render()

        turn_count += 1
        if turn_count > 9:
            print("Draw!")
            break