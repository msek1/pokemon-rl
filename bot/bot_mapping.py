from bot.bot_interface import (
    NetworkBattle, Pokemon, PokemonStats, PokemonMove, PokemonItem,
    PokemonBoosts, PokemonAbility, PokemonStatus)
from poke_env.environment.battle import Battle
import torch
import pickle
from encoding.create_encoding_datasets import PokemonEncodingData, MoveEncodingData, CURRENT_PP_INDEX
from encoding.autoencoder import Autoencoder, EMBEDDING_DIMENSION, DEVICE
from encoding.sphere_encoding import ability_dim, item_dim

BOOST_ENCODING_DIM = 8
STATUS_ENCODING_DIM = 21
TERRAIN_ENCODING_DIM = 20
POKEMON_FULL_ENCODING_DIM = 5*EMBEDDING_DIMENSION + ability_dim + item_dim + STATUS_ENCODING_DIM # Pokemon + Moves have same embedding dim

STATE_DIMENSION = 2*BOOST_ENCODING_DIM + TERRAIN_ENCODING_DIM + 14 * POKEMON_FULL_ENCODING_DIM

class EnvironmentMapper:
    def mapBattle(self, battle: Battle) -> NetworkBattle:
        pass

    def mapSinglePokemon(self):
        pass

class EnvironmentEncoder:
    pokemon_encoder: Autoencoder
    move_encoder: Autoencoder
    item_encoding: dict
    ability_encoding: dict

    move_data: dict

    def __init__(self) -> None:
        # self.pokemon_encoder = Autoencoder(input_dim=44)
        self.pokemon_encoder = torch.load("bot_data/autoencoder_pokemon.pth", weights_only=False).to(DEVICE)
        # self.move_encoder = Autoencoder(input_dim=27)
        self.move_encoder = torch.load("bot_data/autoencoder_moves.pth", weights_only=False).to(DEVICE)
        
        with open("bot_data/constant_encodings.pkl", "rb") as f:
            constant_encodings = pickle.load(f)
        self.item_encoding = constant_encodings["item"]
        self.ability_encoding = constant_encodings["ability"]

        with open("bot_data/move_data.pkl", "rb") as f:
            self.move_data = pickle.load(f)

    def encodeBattle(self, battle: NetworkBattle) -> torch.Tensor:
        # Encode [own active, opp active, both teams, own boost, opp boost, terrain]
        PKMN_DIM = POKEMON_FULL_ENCODING_DIM
        res = torch.zeros(STATE_DIMENSION)
        if battle.ownActive is not None:
            res[:PKMN_DIM] = self.encode_pokemon(battle.ownActive)
            res[14*PKMN_DIM:14*PKMN_DIM+BOOST_ENCODING_DIM] = battle.ownActive.boosts.to_vector()
        if battle.oppActive is not None:
            res[PKMN_DIM:2*PKMN_DIM] = self.encode_pokemon(battle.oppActive)
            res[14*PKMN_DIM+BOOST_ENCODING_DIM:14*PKMN_DIM+2*BOOST_ENCODING_DIM] = battle.ownActive.boosts.to_vector()
        
        for i, pkmn in enumerate(battle.ownPokemon):
            res[(i+2)*PKMN_DIM:(i+3)*PKMN_DIM] = self.encode_pokemon(pkmn)
        for i, pkmn in enumerate(battle.oppPokemon):
            res[(i+8)*PKMN_DIM:(i+9)*PKMN_DIM] = self.encode_pokemon(pkmn)
        
        res[14*PKMN_DIM + 2*BOOST_ENCODING_DIM:] = battle.field.to_vector()
        return res

    def encode_pokemon(self, pokemon: Pokemon) -> torch.Tensor:
        # Encode [pokemon info, move info, ability, item, status]
        result = torch.Tensor(POKEMON_FULL_ENCODING_DIM).to(DEVICE)
        encoding_data = PokemonEncodingData(
            pokemon.types, *pokemon.baseStats.as_list(), pokemon.current_hp
        )
        result[:EMBEDDING_DIMENSION] = self.pokemon_encoder.encoder(encoding_data.to_vector().to(DEVICE))
        
        for i,move in enumerate(pokemon.known_moves):
            v = self.move_data[move.name]
            vec = v.to_vector()
            vec[CURRENT_PP_INDEX] = move.pp_left / v.pp
            result[(i+1) * EMBEDDING_DIMENSION: (i+2)*EMBEDDING_DIMENSION] = self.move_encoder.encoder(vec.to(DEVICE))
        
        ability = "UNKNOWN" if pokemon.ability is None else pokemon.ability.name
        item = "UNKNOWN" if pokemon.item is None else (
            "NONE" if not pokemon.item.holding else pokemon.item.name
        )
        PKMN_MOVE_SIZE = 5*EMBEDDING_DIMENSION
        result[PKMN_MOVE_SIZE:PKMN_MOVE_SIZE + ability_dim] = self.ability_encoding[ability]
        result[PKMN_MOVE_SIZE + ability_dim:PKMN_MOVE_SIZE + ability_dim + item_dim] = self.item_encoding[item]
        result[PKMN_MOVE_SIZE + ability_dim + item_dim:] = pokemon.status.to_vector().to(DEVICE)
        return result
