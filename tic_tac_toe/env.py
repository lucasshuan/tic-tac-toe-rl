class TicTacToeEnv:
    EMPTY = " "
    O = "O"
    X = "X"

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the board and returns the initial state"""
        self.board = [[self.EMPTY for _ in range(3)] for _ in range(3)]
        self.current_player = self.X  # X always starts
        self.turn_count = 1
        return self.get_state()

    def get_state(self):
        """Flattens the board into a tuple for state representation"""
        return tuple(cell for row in self.board for cell in row)

    def available_actions(self):
        """Returns available moves as a list of (x, y) tuples"""
        return [(x, y) for x in range(3) for y in range(3) if self.board[x][y] == self.EMPTY]

    def step(self, action: tuple) -> tuple:
        """Performs an action (placing a marker), then switches turns."""
        x, y = action
        if self.board[x][y] != self.EMPTY:
            raise ValueError(f"Invalid move at ({x}, {y})!")

        # Place the piece
        self.board[x][y] = self.current_player

        # Check for game end
        if self.check_win(self.current_player):
            reward = 1 - (self.turn_count / 9)
            return self.get_state(), reward if self.current_player == self.X else -reward, True
        elif not self.available_actions():
            return self.get_state(), 0, True  # Draw

        # Switch player
        self.current_player = self.O if self.current_player == self.X else self.X
        return self.get_state(), 0, False  # Continue game

    def check_win(self, player):
        """Checks if the given player has won"""
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        if all(self.board[i][i] == player for i in range(3)) or all(self.board[i][2 - i] == player for i in range(3)):
            return True
        return False
    
    def check_win_move(self, player):
        """Check if a player can win in the next move"""
        for x, y in self.available_actions():
            self.board[x][y] = player
            if self.check_win(player):
                self.board[x][y] = self.EMPTY  # Undo the move
                return (x, y)
            self.board[x][y] = self.EMPTY
        return None

    def check_block_move(self, player):
        """Check if the opponent can win in the next move and block it"""
        opponent = self.X if player == self.O else self.O
        return self.check_win_move(opponent)

    def render(self):
        """Prints the current board state"""
        print("")
        print(" ", self.board[0][0], " | ", self.board[0][1], " | ", self.board[0][2])
        print("-----+-----+-----")
        print(" ", self.board[1][0], " | ", self.board[1][1], " | ", self.board[1][2])
        print("-----+-----+-----")
        print(" ", self.board[2][0], " | ", self.board[2][1], " | ", self.board[2][2])
        print("")