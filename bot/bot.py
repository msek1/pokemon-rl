from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.environment.battle import AbstractBattle
from bot.teams import teams
from bot.bot_mapping import EnvironmentMapper, EnvironmentEncoder
from trainer.actorcritic import ActorCritic, create_agent
import torch
import torch.nn.functional as f
from torch.distributions import Categorical
from typing import List, Tuple
import pickle

DEVICE = "cpu"

class RLBot(Player):
    env_mapper: EnvironmentMapper
    env_encoder: EnvironmentEncoder

    battle_data: dict
    name: str
    prev_battle_obs_count: dict

    def __init__(self, name, format, team_name, decision_network: ActorCritic, encoder: EnvironmentEncoder):
        self.name = name
        self.decision_network = decision_network

        team = None if format.endswith("randombattle") else teams[team_name]
        super().__init__(AccountConfiguration(name, None), battle_format=format,  max_concurrent_battles=10, team=team)

        self.env_mapper = EnvironmentMapper()
        self.env_encoder = encoder

        self.battle_data = {} # battle_tag -> List[(state, action, action_log_probability, value)]
        self.prev_battle_obs_count = {}

    def choose_move(self, battle: AbstractBattle):
        mapping = self.env_mapper.mapBattle(battle)
        state = self.env_encoder.encodeBattle(mapping)
        action_scores, value_pred = self.decision_network.forward(state.to(DEVICE))
        action_ind, log_prob, action_mask, order = self.translate_action_scores(action_scores.cpu(), battle)
        
        if battle.battle_tag not in self.prev_battle_obs_count:
            self.prev_battle_obs_count[battle.battle_tag] = None

        if (action_ind != -1):
            if (len(battle.observations) == self.prev_battle_obs_count[battle.battle_tag]):
                if battle.battle_tag in self.battle_data:
                    self.battle_data[battle.battle_tag] = self.battle_data[battle.battle_tag][:len(battle.observations)-1]
            self.add_turn(action_ind, state, value_pred, log_prob, action_mask, battle.battle_tag)
       
        self.prev_battle_obs_count[battle.battle_tag] = len(battle.observations)

        return order

    def write_out_battles(self):
        with open(f"bot_data/tmp/{self.name}_battles.pkl", "wb") as f:
            pickle.dump({"battles": self.battles, "states": self.battle_data}, f)
    
    # def _battle_finished_callback(self, battle: AbstractBattle):
    #     # if abs(len(battle.observations) - len(self.battle_data[battle.battle_tag])) > 1:
    #     #     print(len(battle.observations), len(self.battle_data[battle.battle_tag]), flush=True)
    #     print(battle.battle_tag, len(battle.observations), len(self.battle_data[battle.battle_tag]), flush=True)
    #     assert abs(len(battle.observations) - len(self.battle_data[battle.battle_tag])) <= 1
    
    def add_turn(self, action, state, action_log_prob, value_pred, action_mask, battle_tag):
        if not battle_tag in self.battle_data:
            self.battle_data[battle_tag] = []
        self.battle_data[battle_tag].append((state, action, action_log_prob, action_mask, value_pred))
    
    def get_battle_data(self):
        return self.battle_data
    
    def clear_battle_data(self):
        self.battles.clear()
        self.battle_data.clear()
        self.prev_battle_obs_count.clear()

    def translate_action_scores(self, action_scores: torch.Tensor, battle: AbstractBattle) -> Tuple[int, torch.Tensor, torch.Tensor, BattleOrder]:
        available_inds = []
        
        available_moves = [move.id for move in battle.available_moves]
        if battle.active_pokemon is not None:
            for (i,move) in enumerate(battle.active_pokemon.moves.values()):
                if move.id in available_moves:
                    available_inds.append(i)
        
        # available_zs = [] if not battle.can_z_move else [4 + i for i in available_inds]
        available_zs = []
        available_megas = [] if not battle.can_mega_evolve else [8 + i for i in available_inds]

        team_non_active = [pkmn.base_species for pkmn in battle.team.values() if not pkmn.active]
        switches = battle.available_switches
        switch_inds = []
        try:
            switch_inds = [12 + i for i in range(5) if i < len(switches) and i == team_non_active.index(switches[i].base_species)]
        except Exception as e:
            print("FAILED_SWITCHES:", team_non_active, switches)
        possible_inds = torch.Tensor(available_inds + available_zs + available_megas + switch_inds).long()

        if len(possible_inds) == 0:
            order = super().choose_random_singles_move(battle)
            return -1, None, torch.zeros(17), order

        applicable_scores = action_scores[possible_inds]
        probs = f.softmax(applicable_scores, dim=0)
        dist = Categorical(probs)
        chosen_action = dist.sample()
        action_ind = possible_inds[chosen_action.item()]
        # log_prob_action = dist.log_prob(chosen_action)
        log_prob_action = Categorical(f.softmax(action_scores, dim=0)).log_prob(action_ind)

        if action_ind < 4:
            order = BattleOrder(list(battle.active_pokemon.moves.values())[action_ind])
        elif action_ind < 8:
            order = BattleOrder(list(battle.active_pokemon.moves.values())[action_ind-4], z_move=True)
        elif action_ind < 12:
            order = BattleOrder(list(battle.active_pokemon.moves.values())[action_ind-8], mega=True)
        else:
            order = BattleOrder(battle.available_switches[action_ind - 12])
        
        mask = torch.zeros(17)
        mask.scatter_(0,possible_inds,1)
        return action_ind, log_prob_action, mask, order

