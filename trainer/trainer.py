from poke_env.player import Player
import asyncio

class Trainer:
    async def make_players_play(self, p1: Player, p2: Player, num_battles: int):
        await asyncio.gather(
            p1.accept_challenges(None, num_battles),
            p2.send_challenges("pokemonrlbotcs486", num_battles)
        )
        