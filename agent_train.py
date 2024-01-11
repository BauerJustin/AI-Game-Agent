from src.agent import AlphaZero, ResNet
from src.games import TicTacToe, ConnectFour
import torch


def main():
    game = TicTacToe()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 4, 64, device)

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
