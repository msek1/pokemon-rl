from bot.bot import RLBot
from encoding.autoencoder import Autoencoder
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
    asyncio.run(main())

'''
if __name__ == "__main__":
    bot = RLBot("pokemonrlbotcs486", "gen7ou", "gen_7_team")
    asyncio.run(bot.play())
'''