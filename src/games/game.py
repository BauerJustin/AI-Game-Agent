import numpy as np


class Game:
    def __init__(self, row_count, column_count, action_size):
        self.row_count = row_count
        self.column_count = column_count
        self.action_size = action_size

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        raise NotImplementedError("Subclasses must implement this method")

    def get_valid_moves(self, state):
        raise NotImplementedError("Subclasses must implement this method")

    def check_win(self, state, action):
        raise NotImplementedError("Subclasses must implement this method")

    def get_value_and_terminated(self, state, action):
        raise NotImplementedError("Subclasses must implement this method")

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
