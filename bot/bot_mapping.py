from bot.bot_interface import (
    NetworkBattle, Pokemon, PokemonStats, PokemonMove, PokemonItem,
    PokemonBoosts, PokemonAbility, PokemonStatus, PokemonEphemeralStatus, BattleTerrain)
from poke_env.environment.battle import AbstractBattle
from poke_env.environment import Effect, SideCondition, Weather, Field, Pokemon as PokeEnvPkmn, Status
import torch
import pickle
from encoding.create_encoding_datasets import PokemonEncodingData, CURRENT_PP_INDEX, get_log_effectiveness_vector
from encoding.autoencoder import Autoencoder, EMBEDDING_DIMENSION
from encoding.sphere_encoding import ability_dim, item_dim
from typing import List, Optional

BOOST_ENCODING_DIM = 8
STATUS_ENCODING_DIM = 8
EPHEMERAL_STATUS_DIM = 15
MEGA_Z_DIM = 4
TERRAIN_ENCODING_DIM = 32
POKEMON_FULL_ENCODING_DIM = 5*EMBEDDING_DIMENSION + ability_dim + item_dim + STATUS_ENCODING_DIM # Pokemon + Moves have same embedding dim

STATE_DIMENSION = 2*BOOST_ENCODING_DIM + TERRAIN_ENCODING_DIM + 14 * POKEMON_FULL_ENCODING_DIM + 2*EPHEMERAL_STATUS_DIM + MEGA_Z_DIM

DEVICE = "cpu"

class EnvironmentMapper:
    def mapBattle(self, battle: AbstractBattle) -> NetworkBattle:
        ownTeam = battle.team.values()
        oppTeam = list(battle.opponent_team.values())
        oppTeamSpecies = set([p.base_species for p in oppTeam])
        for p in battle.teampreview_opponent_team:
            if p.base_species not in oppTeamSpecies:
                oppTeam.append(p)

        ownPkmn = list(map(lambda p: self.mapSinglePokemon(p), ownTeam))
        oppPkmn = list(map(lambda p: self.mapSinglePokemon(p), oppTeam))

        ownActive = None if battle.active_pokemon is None else self.mapSinglePokemon(battle.active_pokemon)
        oppActive = None if battle.opponent_active_pokemon is None else self.mapSinglePokemon(battle.opponent_active_pokemon)

        ownEffects = None if battle.active_pokemon is None else self.mapEphemeralEffects(battle.active_pokemon)
        oppEffects = None if battle.opponent_active_pokemon is None else self.mapEphemeralEffects(battle.opponent_active_pokemon)

        return NetworkBattle(
            ownPkmn, ownActive, ownEffects,
            oppPkmn, oppActive, oppEffects,
            self.mapTerrain(battle),
            battle.can_z_move, battle.can_mega_evolve,
            battle._opponent_can_z_move, battle._opponent_can_mega_evolve
        )


    def mapSinglePokemon(self, pokemon: PokeEnvPkmn) -> Pokemon:
        stats = PokemonStats(
            pokemon.base_stats['atk'], pokemon.base_stats['def'], pokemon.base_stats['spa'],
            pokemon.base_stats['spd'], pokemon.base_stats['spe'], pokemon.base_stats['hp'],
        )

        return Pokemon(
            name = pokemon.base_species,
            baseStats=stats,
            stats = None,
            boosts=self.mapBoosts(pokemon),
            types = [t.name.capitalize() for t in pokemon.types],
            known_moves=self.mapMoves(pokemon),
            ability=self.mapAbility(pokemon),
            item = self.mapItem(pokemon),
            status=self.mapStatus(pokemon),
            current_hp=pokemon.current_hp_fraction
        )


    def mapAbility(self, pokemon: PokeEnvPkmn) -> Optional[PokemonAbility]:
        if pokemon.ability is None:
            return None
        return PokemonAbility(pokemon.ability)
    
    def mapItem(self, pokemon: PokeEnvPkmn) -> Optional[PokemonItem]:
        if pokemon.item is None:
            return PokemonItem(holding=False)
        elif pokemon.item == "unknown_item":
            return None
        return PokemonItem(pokemon.item, holding=True)

    def mapMoves(self, pokemon: PokeEnvPkmn) -> List[PokemonMove]:
        return [PokemonMove(mv.id, mv.current_pp) for mv in pokemon.moves.values()]

    def mapStatus(self, pokemon: PokeEnvPkmn) -> PokemonStatus:
        res = PokemonStatus()
        if pokemon.status is Status.PSN:
            res.majorStatus = "poison"
        elif pokemon.status is Status.TOX:
            res.majorStatus = "toxic"
        elif pokemon.status is Status.PAR:
            res.majorStatus = "paralysis"
        elif pokemon.status is Status.SLP:
            res.majorStatus = "sleep"
        elif pokemon.status is Status.BRN:
            res.majorStatus = "burn"
        elif pokemon.status is Status.FRZ:
            res.majorStatus = "freeze"
        elif pokemon.status is Status.FNT:
            res.majorStatus = "faint"

        if pokemon.status is not None:
            res.statusTurns = pokemon.status_counter
        
        return res

    def mapBoosts(self, pokemon: PokeEnvPkmn) -> PokemonStatus:
        boosts = pokemon.boosts
        
        return PokemonBoosts(
            boosts['atk'], boosts['def'], boosts['spa'], boosts['spd'], boosts['spe'],
            boosts['accuracy'], boosts['evasion']
        )

    def mapEphemeralEffects(self, pokemon: PokeEnvPkmn) -> PokemonEphemeralStatus:
        res = PokemonEphemeralStatus()
        for eff in pokemon.effects:
            if eff is Effect.CONFUSION:
                res.isConfused = True
            elif eff is Effect.ATTRACT:
                res.isInfatuated = True
            elif eff is Effect.YAWN:
                res.isDrowsy = True
            elif eff is Effect.SUBSTITUTE:
                res.isSubstituted = True
            elif eff is Effect.PERISH3:
                if res.perishSongTurns == 0:
                    res.perishSongTurns = 3
            elif eff is Effect.PERISH2:
                if res.perishSongTurns == 0 or res.perishSongTurns > 2:
                    res.perishSongTurns = 2
            elif eff is Effect.PERISH1:
                if res.perishSongTurns == 0 or res.perishSongTurns > 1:
                    res.perishSongTurns = 1
            elif eff is Effect.LEECH_SEED:
                res.isSeeded = True
            elif eff is Effect.DISABLE:
                res.disabledMove = 0
            elif eff is Effect.TRAPPED:
                res.isTrapped = True
            elif eff is Effect.TAUNT:
                res.isTaunted = True
            elif eff is Effect.TORMENT:
                res.isTormented = True
        
        if pokemon.protect_counter > 0:
            res.didProtect = True
        if pokemon.must_recharge:
            res.isRecharging = True
        if pokemon.preparing_move:
            res.isCharging = True
        if pokemon.first_turn:
            res.isFirstTurn = True
        
        return res

    def mapTerrain(self, battle: AbstractBattle) -> BattleTerrain:
        res = BattleTerrain()

        for w in battle.weather:
            if w is Weather.SUNNYDAY or w is Weather.DESOLATELAND:
                res.weather = "sun"
            elif w is Weather.RAINDANCE or w is Weather.PRIMORDIALSEA:
                res.weather = "rain"
            elif w is Weather.HAIL:
                res.weather = "snow"
            elif w is Weather.SANDSTORM:
                res.weather = "sandstorm"
            
            res.weatherTurns = battle.weather[w]

        for f in battle.fields:
            if f is Field.ELECTRIC_TERRAIN:
                res.terrain = "electric"
                res.terrainTurns = f.value
            elif f is Field.GRASSY_TERRAIN:
                res.terrain = "grassy"
                res.terrainTurns = f.value
            elif f is Field.PSYCHIC_TERRAIN:
                res.terrain = "psychic"
                res.terrainTurns = f.value
            elif f is Field.MISTY_TERRAIN:
                res.terrain = "misty"
                res.terrainTurns = f.value
            
            elif f is Field.TRICK_ROOM:
                res.isTrickRoom = f.value
            elif f is Field.MAGIC_ROOM:
                res.isMagicRoom = f.value
            elif f is Field.WONDER_ROOM:
                res.isWonderRoom = f.value
            elif f is Field.GRAVITY:
                res.isGravity = f.value

        for sc in battle.side_conditions:
            if sc is SideCondition.SPIKES:
                res.teamSpikesLayers = sc.value
            elif sc is SideCondition.TOXIC_SPIKES:
                res.teamPoisonLayers = sc.value
            elif sc is SideCondition.TAILWIND:
                res.teamTailwind = sc.value
            elif sc is SideCondition.SAFEGUARD:
                res.teamSafeguard = sc.value
            elif sc is SideCondition.STEALTH_ROCK:
                res.teamStealthRocks = True
            elif sc is SideCondition.STICKY_WEB:
                res.teamStickyWeb = True
            elif sc is SideCondition.AURORA_VEIL:
                res.teamAuroraVeil = True
            elif sc is SideCondition.REFLECT:
                res.teamReflect = True
            elif sc is SideCondition.LIGHT_SCREEN:
                res.teamLightScreen = True
        
        for sc in battle.opponent_side_conditions:
            if sc is SideCondition.SPIKES:
                res.oppSpikesLayers = sc.value
            elif sc is SideCondition.TOXIC_SPIKES:
                res.oppPoisonLayers = sc.value
            elif sc is SideCondition.TAILWIND:
                res.oppTailwind = sc.value
            elif sc is SideCondition.SAFEGUARD:
                res.oppSafeguard = sc.value
            elif sc is SideCondition.STEALTH_ROCK:
                res.oppStealthRocks = True
            elif sc is SideCondition.STICKY_WEB:
                res.oppStickyWeb = True
            elif sc is SideCondition.AURORA_VEIL:
                res.oppAuroraVeil = True
            elif sc is SideCondition.REFLECT:
                res.oppReflect = True
            elif sc is SideCondition.LIGHT_SCREEN:
                res.oppLightScreen = True
        return res
            
        

class EnvironmentEncoder:
    pokemon_encoder: Autoencoder
    move_encoder: Autoencoder
    item_encoding: dict
    ability_encoding: dict

    move_data: dict

    def __init__(self) -> None:
        # self.pokemon_encoder = Autoencoder(input_dim=44)
        self.pokemon_encoder = torch.load("bot_data/autoencoder_pokemon.pth", weights_only=False, map_location=torch.device(DEVICE))
        # self.move_encoder = Autoencoder(input_dim=27)
        self.move_encoder = torch.load("bot_data/autoencoder_moves.pth", weights_only=False, map_location=torch.device(DEVICE))
        
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

        PKMN_AND_BOOSTS = 14*PKMN_DIM + 2*BOOST_ENCODING_DIM
        
        if battle.ownActiveStatus is not None:
            res[PKMN_AND_BOOSTS: PKMN_AND_BOOSTS + EPHEMERAL_STATUS_DIM] = battle.ownActiveStatus.to_vector()
        if battle.oppActiveStatus is not None:
            res[PKMN_AND_BOOSTS + EPHEMERAL_STATUS_DIM: PKMN_AND_BOOSTS + 2*EPHEMERAL_STATUS_DIM] = battle.oppActiveStatus.to_vector()
        
        res[PKMN_AND_BOOSTS + 2*EPHEMERAL_STATUS_DIM : PKMN_AND_BOOSTS + 2*EPHEMERAL_STATUS_DIM + TERRAIN_ENCODING_DIM] = battle.field.to_vector()
        PRE_MOVE_MODIFIER_DIM = PKMN_AND_BOOSTS + 2*EPHEMERAL_STATUS_DIM + TERRAIN_ENCODING_DIM
        res[PRE_MOVE_MODIFIER_DIM:] = torch.Tensor([battle.can_z_move, battle.can_mega, battle.can_opp_z_move, battle.can_opp_mega])
        return res.to(DEVICE)

    def encode_pokemon(self, pokemon: Pokemon) -> torch.Tensor:
        # Encode [pokemon info, move info, ability, item, status]
        result = torch.zeros(POKEMON_FULL_ENCODING_DIM)
        if pokemon is None:
            return result
        encoding_data = PokemonEncodingData(
            pokemon.types, *pokemon.baseStats.as_list(), pokemon.current_hp
        )
        result[:EMBEDDING_DIMENSION] = self.pokemon_encoder.encoder(encoding_data.to_vector().to(DEVICE))
        
        for i,move in enumerate(pokemon.known_moves):
            if move.name.startswith("hiddenpower") and len(move.name) > 11:
                hp_type = move.name[11:].capitalize()
                effectiveness = get_log_effectiveness_vector(hp_type)
                v = self.move_data["hiddenpower"]
                v.effectiveness = torch.Tensor(effectiveness)
            else:
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
        if item != "":
            result[PKMN_MOVE_SIZE + ability_dim:PKMN_MOVE_SIZE + ability_dim + item_dim] = self.item_encoding[item]
        else:
            # print(pokemon)
            pass
        result[PKMN_MOVE_SIZE + ability_dim + item_dim:] = pokemon.status.to_vector().to(DEVICE)
        return result
