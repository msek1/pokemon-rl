from poke_env import AccountConfiguration, ServerConfiguration
from bot.bot import RLBot
import bot.teams as teams
from encoding.autoencoder import Autoencoder
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
import matplotlib.pyplot as plt

from trainer.actorcritic import create_agent
from trainer.trainer import Trainer
import asyncio
import torch.optim as optim

if __name__ == "__main__":
    NETWORK_HIDDEN_DIM = 512
    NETWORK_DROPOUT = 0.2
    LEARNING_RATE = 0.001
    decision_network = create_agent(NETWORK_HIDDEN_DIM, NETWORK_DROPOUT)
    
    bot = RLBot("pokemonrlbotcs486", "gen7ou", "gen_7_team", decision_network)
    
    opp = SimpleHeuristicsPlayer(AccountConfiguration("rlbotopp", None), battle_format="gen7ou",  max_concurrent_battles=100, team=teams.teams["gen_7_team"])
    
    trainer = Trainer(battles_per_epoch=200)
    # asyncio.run(trainer.make_players_play(bot, opp, 20))

    optimizer = optim.Adam(decision_network.parameters(), lr=LEARNING_RATE)
    # policy_loss, value_loss = trainer.run_epoch(bot, opp, optimizer)
    # print(policy_loss, value_loss)

    losses,evals = trainer.run_epochs(bot, opp, optimizer, 10)

    plt.plot([l[0] for l in losses])
    plt.show()
