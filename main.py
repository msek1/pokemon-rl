from poke_env import AccountConfiguration, ServerConfiguration
from bot.bot import RLBot
import bot.teams as teams
from encoding.autoencoder import Autoencoder
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from trainer.trainer import Trainer
import asyncio


if __name__ == "__main__":
    custom_config = None
    bot = RLBot("pokemonrlbotcs48", "gen7ou", "gen_7_team")
    opp = RandomPlayer(AccountConfiguration("rlbotopp", None), server_configuration=custom_config, battle_format="gen7ou",  max_concurrent_battles=100000, team=teams.teams["gen_7_team"])
    
    trainer = Trainer()
    asyncio.run(trainer.make_players_play(bot, opp, 1))
    print(bot.battles)

    for (k, v) in bot.battle_data.items():
        # each battle
        print(v[0])
