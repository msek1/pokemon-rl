from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.environment.battle import AbstractBattle
from bot.teams import teams
from bot.bot_mapping import EnvironmentMapper, EnvironmentEncoder
from encoding.autoencoder import DEVICE

import time

class RLBot(SimpleHeuristicsPlayer):
    env_mapper: EnvironmentMapper
    env_encoder: EnvironmentEncoder

    def __init__(self, name, format, team_name):
        super().__init__(AccountConfiguration(name, None), battle_format=format,  max_concurrent_battles=100000, team=teams[team_name])

        self.env_mapper = EnvironmentMapper()
        self.env_encoder = EnvironmentEncoder()

    def choose_move(self, battle: AbstractBattle):
        start = time.time_ns()
        mapping = self.env_mapper.mapBattle(battle)
        mid = time.time_ns()
        encoding = self.env_encoder.encodeBattle(mapping)
        end = time.time_ns()
        print(f"Mapping Time: {(mid - start) / 1000000}ms, {(end - mid) / 1000000}ms")
        print(encoding.device)
        return super().choose_move(battle)

    async def play(self):
        # No authentication required
        # my_account_config = AccountConfiguration("pokemonrlbotcs486", None)
        # player = SimpleHeuristicsPlayer(account_configuration=my_account_config, battle_format="gen9ou", team=ou_team)
        # player = SimpleHeuristicsPlayer(account_configuration=my_account_config, battle_format="gen7ou", team=gen_7_team)
        # player = SimpleHeuristicsPlayer(account_configuration=my_account_config, battle_format="gen7anythinggoes", team=teams["gen_7_ag"])
        # player = RandomPlayer(account_configuration=my_account_config, battle_format="gen7anythinggoes", team=gen_7_ag)
        # player = SimpleHeuristicsPlayer(account_configuration=my_account_config)
        # team = Teambuilder.parse_showdown_team(ou_team)
        print("Starting bot...")
        # print(str(team)[1:-1])
        await self.accept_challenges(None, 1)
