from src.agent import AlphaZero, ResNet
from src.games import TicTacToe, ConnectFour
from config import GAME, NUM_RES_BLOCKS, NUM_HIDDEN_UNITS
import torch


def main():
    if GAME == "TicTacToe":
        game = TicTacToe()
    elif GAME == "ConnectFour":
        game = ConnectFour()
    else:
        raise Exception(f"Invalid GAME type: {GAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, NUM_RES_BLOCKS, NUM_HIDDEN_UNITS, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 1,
        'num_selfPlay_iterations': 50,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()

if __name__ == "__main__":
    main()
