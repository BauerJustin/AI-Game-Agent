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


class TicTacToe(Game):
    def __init__(self):
        super().__init__(row_count=3, column_count=3, action_size=9)

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False


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
