from alpha_zero import AlphaZero
from src.game import TicTacToe, ConnectFour
from src.model import ResNet
import torch


def main():
    game = TicTacToe()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device)

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
