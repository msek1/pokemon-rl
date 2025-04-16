from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
import asyncio

async def play():
    # No authentication required
    my_account_config = AccountConfiguration("pokemonrlbotcs486", None)
    player = SimpleHeuristicsPlayer(account_configuration=my_account_config)
    print("Starting bot...")
    await player.accept_challenges(None, 1)

if __name__ == "__main__":
    asyncio.run(play())
