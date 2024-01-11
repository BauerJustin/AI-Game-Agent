from src.games import TicTacToe, ConnectFour
from src.agent import ResNet, MCTS
from config import GAME
import numpy as np
import torch


def main():
    if GAME == "TicTacToe":
        game = TicTacToe()
    elif GAME == "ConnectFour":
        game = ConnectFour()
    else:
        raise Exception(f"Invalid GAME type: {GAME}")
    player = 1

    args = {
        'C': 2,
        'num_searches': 100,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 4, 64, device)
    model.load_state_dict(torch.load(f"models/model_{GAME}.pt", map_location=device))
    model.eval()

    mcts = MCTS(game, args, model)

    state = game.get_initial_state()

    while True:
        print(state)
        
        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("Moves: ", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print(f"Invalid move: {action}")
                continue
                
        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            
        state = game.get_next_state(state, action, player)
        
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            print(state)
            if value == 1:
                print(player, "Won!")
            else:
                print("Draw")
            break
            
        player = game.get_opponent(player)

if __name__ == "__main__":
    main()
