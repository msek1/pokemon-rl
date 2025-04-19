from poke_env import AccountConfiguration
from bot.bot import RLBot
import bot.teams as teams
from encoding.autoencoder import Autoencoder
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from trainer.trainer import Trainer
import asyncio

async def main():
    format = "gen7randombattle"
    team_name = "gen_7_team"

    receiver = RLBot("receiver_bot", format, team_name)
    sender = RLBot("sender_bot", format, team_name)

    await asyncio.gather(
        receiver.accept_challenges(None, 1),
        sender.send_challenges("receiver_bot", n_challenges=1)
    )

    for battle in sender.battles.values():
        print("\n".join(battle.battle_log))

if __name__ == "__main__":
    bot = RLBot("pokemonrlbotcs486", "gen7ou", "gen_7_team")
    opp = RandomPlayer(AccountConfiguration("rlbotopp", None), battle_format="gen7ou",  max_concurrent_battles=100000, team=teams.teams["gen_7_team"])
    
    trainer = Trainer()
    asyncio.run(trainer.make_players_play(bot, opp, 200))
    print(bot.battles.values())
