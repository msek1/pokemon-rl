from poke_env import AccountConfiguration, ServerConfiguration
from bot.bot import RLBot
import bot.teams as teams
from bot.bot_mapping import EnvironmentEncoder
from encoding.autoencoder import Autoencoder
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
import matplotlib.pyplot as plt
import pickle

from trainer.actorcritic import create_agent
from trainer.trainer import Trainer
import asyncio
import torch
import torch.optim as optim

if __name__ == "__main__":
    NETWORK_HIDDEN_DIM = 512
    NETWORK_DROPOUT = 0.2
    LEARNING_RATE = 0.001
    decision_network = create_agent(NETWORK_HIDDEN_DIM, NETWORK_DROPOUT)

    torch.save(decision_network, "bot_data/main_model.pth")

    trainer = Trainer(battles_per_epoch=50)
    optimizer = optim.Adam(decision_network.parameters(), lr=LEARNING_RATE)
    losses,evals = trainer.run_epochs(decision_network, optimizer, 10, checkpoint_interval=5)

    with open("training_results.pkl", "wb") as f:
        pickle.dump({"loss": losses, "eval": evals}, f)
    
    print([l[0] for l in losses])
    plt.plot([l[0] for l in losses])
    plt.show()
