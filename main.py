from bot.bot import RLBot
from encoding.autoencoder import Autoencoder
import asyncio

if __name__ == "__main__":
    bot = RLBot("pokemonrlbotcs486", "gen7ou", "gen_7_team")
    asyncio.run(bot.play())