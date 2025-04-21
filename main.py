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
from copy import deepcopy

if __name__ == "__main__":
    NETWORK_HIDDEN_DIM = 512
    NETWORK_DROPOUT = 0.2
    LEARNING_RATE = 0.0001
    decision_network = create_agent(NETWORK_HIDDEN_DIM, NETWORK_DROPOUT).cpu()

    torch.save(decision_network, "bot_data/main_model.pth")

    trainer = Trainer(battles_per_epoch=25)
    optimizer = optim.Adam(decision_network.parameters(), lr=LEARNING_RATE)
    encoder = EnvironmentEncoder()
    bot = RLBot("rlbotcs486", "gen7randombattle", None, decision_network, encoder)
    opp = RLBot("rloppcs486", "gen7randombattle", None, decision_network, encoder)

    losses, evals, steps = trainer.run_epochs(bot, opp, optimizer, 3000, set_weights_interval=25, checkpoint_interval=25, evaluation_interval=25)

    with open("training_results.pkl", "wb") as f:
        pickle.dump({"loss": losses, "eval": evals, "steps": steps}, f)
    
    print([l[0] for l in losses])
    plt.plot([l[0] for l in losses])
    plt.show()
