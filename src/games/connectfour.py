import numpy as np
from .game import Game


class ConnectFour(Game):
    def __init__(self):
        super().__init__(row_count=6, column_count=7, action_size=7)
        self.in_a_row = 4

    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count_direction(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r, c = row + offset_row * i, column + offset_column * i
                if not (0 <= r < self.row_count and 0 <= c < self.column_count) or state[r, c] != player:
                    return i
            return self.in_a_row

        def check_line(offset_row, offset_column):
            return count_direction(offset_row, offset_column) + count_direction(-offset_row, -offset_column) >= self.in_a_row

        return check_line(1, 0) or check_line(0, 1) or check_line(1, 1) or check_line(1, -1)

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        elif np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        else:
            return 0, False
